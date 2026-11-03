from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import geostat.settings as settings


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_hex(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_b64(path: Path) -> str:
    import base64
    return base64.b64encode(bytes.fromhex(md5_hex(path))).decode()


def _source_files(root: Path | None = None) -> list[Path]:
    root = root or settings.PROJECT_ROOT
    files: list[Path] = []
    for sub in ["src"]:
        base = root / sub
        if base.exists():
            files.extend(sorted(p for p in base.rglob("*.py") if p.is_file()))
    extra = root / "pyproject.toml"
    if extra.exists():
        files.append(extra)
    return sorted(set(files))


def code_revision(root: Path | None = None) -> dict:
    root = root or settings.PROJECT_ROOT
    info: dict = {}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10
        )
        if head.returncode == 0:
            info["git_head"] = head.stdout.strip()[:12]
            dirty = [ln for ln in status.stdout.splitlines() if ln.strip()]
            info["git_dirty"] = bool(dirty)
            info["git_dirty_count"] = len(dirty)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    h = hashlib.sha256()
    for p in _source_files(root):
        rel = p.relative_to(root).as_posix()
        h.update(f"{rel}\n{sha256_hex(p)}\n".encode())
    info["source_tree_sha256"] = h.hexdigest()
    return info


def _ram_gb() -> float | None:
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return round(stat.ullTotalPhys / (1024 ** 3), 2)
    except Exception:
        return None


def _ram_gb_posix() -> float | None:
    try:
        return round(open("/proc/meminfo").readline().split()[1] and 0)
    except Exception:
        return None


def environment_snapshot() -> dict:
    import os

    env: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "cpu_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gdal": _gdal_version(),
        "torch": _torch_version(),
    }
    return env


def _gdal_version() -> str:
    try:
        import rasterio

        return rasterio.__gdal_version__
    except Exception:
        return "unknown"


def _torch_version() -> dict:
    try:
        import torch

        return {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": (torch.version.cuda if hasattr(torch.version, "cuda") else None),
        }
    except Exception:
        return {"version": "absent", "cuda_available": False, "cuda_version": None}
