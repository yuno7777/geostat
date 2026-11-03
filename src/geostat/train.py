from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from geostat import db, metrics
from geostat.dataset import FloodTileDataset
from geostat.settings import ARTIFACTS_DIR
from geostat.unet import SmallUNet, masked_bce_dice, count_params

TRAIN_CONFIG_DEFAULTS = {
    "manifest": None,
    "model_kind": "unet",
    "channels": [8, 16, 32, 64],
    "epochs": 12,
    "batch_size": 16,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "seed": 42,
    "patience": 4,
    "threshold": 0.5,
    "num_workers": 0,
    "torch_threads": 8,
    "augment": True,
}


class TrainError(RuntimeError):
    pass


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    conn: Any,
    run: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    torch.set_num_threads(int(config.get("torch_threads", 8)))
    _seed_everything(int(config["seed"]))
    manifest_path = Path(config["manifest"])
    if not manifest_path.exists():
        raise TrainError(f"manifest not found: {manifest_path}")
    from geostat import geoio

    manifest = geoio.load_json(manifest_path)
    run.log(f"manifest: {manifest_path} tiles={manifest['count']}")

    channels = tuple(int(c) for c in config.get("channels", [8, 16, 32, 64]))
    model = SmallUNet(in_ch=2, out_ch=1, channels=channels)
    run.log(f"model params: {count_params(model)}")

    train_ds = FloodTileDataset(manifest, "train", augment=bool(config.get("augment", True)), seed=int(config["seed"]))
    val_ds = FloodTileDataset(manifest, "val", return_metadata=True)
    batch_size = int(config["batch_size"])
    gen = torch.Generator()
    gen.manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=int(config.get("num_workers", 0)), generator=gen
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(config["epochs"]))
    pos_weight = _train_pos_weight(train_ds)
    run.log(f"pos_weight: {pos_weight:.4f}")

    best_val_iou = -1.0
    best_state: Optional[dict] = None
    best_epoch = -1
    epochs = int(config["epochs"])
    patience = int(config["patience"])
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for xb, yb, mb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = masked_bce_dice(logits, yb, mb, pos_weight=pos_weight)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        val_metrics = _eval_split(model, val_ds, threshold=float(config["threshold"]))
        val_iou = float(val_metrics["iou"])
        history.append({"epoch": epoch, "loss_mean": float(np.mean(losses)), "loss_min": float(np.min(losses)), "loss_max": float(np.max(losses)), "val": val_metrics, "seconds": round(time.time() - t0, 2)})
        run.log(f"epoch {epoch}/{epochs} loss={np.mean(losses):.4f} val_iou={val_iou:.4f} val_f1={val_metrics['f1']:.4f} ({history[-1]['seconds']}s)")
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if best_epoch > 0 and epoch - best_epoch >= patience:
            run.log(f"early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break
    if best_state is None:
        run.log("model never improved on validation; using final weights", level="warn")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_epoch = epochs
    model.load_state_dict(best_state)

    model_dir = ARTIFACTS_DIR / "models" / run.run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / "model.pth"
    torch.save({"state_dict": best_state, "channels": list(channels), "config": config, "best_val_iou": best_val_iou, "best_epoch": best_epoch}, weights_path)
    from geostat.provenance import sha256_hex

    weights_sha = sha256_hex(weights_path)
    summary = {"model": "unet", "channels": list(channels), "epochs": epochs, "best_epoch": best_epoch, "best_val_iou": best_val_iou, "params": count_params(model), "weights_sha256": weights_sha, "history": history, "nondeterminism_note": "CPU training; DataLoader shuffling seeded; no CUDA nondeterminism applies."}
    run.add_artifact("model_weights", weights_path)
    run.add_artifact("model_meta", model_dir / "model.json")
    geoio.save_json(model_dir / "model.json", summary)
    run.set_metrics({"best_val_iou": best_val_iou, "best_epoch": best_epoch, "epochs": epochs})
    train_ds.close()
    val_ds.close()
    return summary


def _train_pos_weight(train_ds: FloodTileDataset) -> float:
    water = 0
    valid = 0
    step = max(1, len(train_ds) // 64)
    for idx in range(0, len(train_ds), step):
        x, y, m = train_ds[idx]
        yv = y[m > 0.5]
        valid += int(yv.numel())
        water += int(yv.sum().item())
    neg = max(1, valid - water)
    return float(max(1.0, neg / max(1, water)))


@torch.no_grad()
def _eval_split(model: Any, ds: FloodTileDataset, threshold: float = 0.5) -> dict[str, Any]:
    model.eval()
    entries = []
    for item in ds:
        x = item["x"][None]
        y = item["y"].numpy()
        m = item["mask"].numpy()
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].numpy()
        pred = (prob >= threshold).astype(np.uint8)
        entries.append({
            "event": item["event"],
            "counts": metrics.confusion_counts(pred, y.astype(np.uint8), m.astype(np.uint8)),
        })
    agg = metrics.aggregate([e["counts"] for e in entries])
    out = dict(agg)
    out["per_event"] = metrics.per_event_metrics(entries)
    out["n_tiles"] = len(entries)
    return out


@torch.no_grad()
def evaluate_model(
    conn: Any,
    run: Any,
    manifest_path: Path,
    weights_path: Path,
    threshold: float,
    split: str = "test",
    model_id: Optional[str] = None,
) -> dict[str, Any]:
    torch.set_num_threads(8)
    from geostat import geoio

    manifest = geoio.load_json(manifest_path)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    channels = tuple(int(c) for c in state["channels"])
    model = SmallUNet(in_ch=2, out_ch=1, channels=channels)
    model.load_state_dict(state["state_dict"])
    ds = FloodTileDataset(manifest, split, return_metadata=True)
    res = _eval_split(model, ds, threshold)
    res["threshold"] = threshold
    res["split"] = split
    res["n_scenes"] = manifest["count"].get("scenes")
    run.log(f"eval split={split} iou={res['iou']:.4f} f1={res['f1']:.4f} prec={res['precision']:.4f} recall={res['recall']:.4f} n={res['n_valid']}")
    for ev, m in res["per_event"].items():
        run.log(f"  event={ev} iou={m['iou']:.4f} f1={m['f1']:.4f} prevalence={m['prevalence']:.4f} n_valid={m['n_valid']}")
    return res
