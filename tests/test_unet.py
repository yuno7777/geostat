from __future__ import annotations

import torch

from geostat.unet import SmallUNet, count_params, masked_bce_dice


def test_unet_shapes_and_forward():
    model = SmallUNet(in_ch=2, out_ch=1, channels=(8, 16, 32, 64))
    x = torch.randn(2, 2, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_unet_backward_and_gradients():
    model = SmallUNet(in_ch=2, out_ch=1, channels=(8, 16, 32, 64))
    x = torch.randn(2, 2, 64, 64)
    y = torch.randint(0, 2, (2, 64, 64)).float()
    mask = torch.ones((2, 64, 64), dtype=torch.bool)
    # Mask out some pixels
    mask[:, 0:10, 0:10] = False

    logits = model(x)
    loss = masked_bce_dice(logits, y, mask, pos_weight=1.5)
    assert loss.item() > 0
    loss.backward()

    # Verify gradients computed
    for p in model.parameters():
        assert p.grad is not None


def test_count_params():
    model = SmallUNet(in_ch=2, out_ch=1, channels=(8, 16, 32, 64))
    n = count_params(model)
    assert 100_000 < n < 150_000
