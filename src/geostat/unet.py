from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 4) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch) if out_ch % groups == 0 else 1, num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch) if out_ch % groups == 0 else 1, num_channels=out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 1, channels: Sequence[int] = (8, 16, 32, 64)) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.enc1 = ConvBlock(in_ch, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)
        self.head = nn.Conv2d(c1, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

def masked_bce_dice(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    *,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    pos_weight: Optional[torch.Tensor | float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    mask3 = mask[:, None, :, :].float()
    bce = F.binary_cross_entropy_with_logits(
        logits, y[:, None, :, :].float(), weight=mask3, pos_weight=pos_weight, reduction="sum"
    ) / (mask3.sum() + eps)
    probs = torch.sigmoid(logits)
    probs_m = probs * mask3
    y_m = y[:, None, :, :].float() * mask3
    dims = (1, 2, 3)
    inter = (probs_m * y_m).sum(dim=dims)
    denom = probs_m.sum(dim=dims) + y_m.sum(dim=dims)
    dice = 1.0 - ((2.0 * inter + eps) / (denom + eps))
    return bce_weight * bce + dice_weight * dice.mean()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
