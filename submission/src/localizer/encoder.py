from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SiameseEncoder(nn.Module):
    def __init__(self, out_channels: int = 128):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.body = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2
        )
        for mod in self.body[5].modules():
            if isinstance(mod, nn.Conv2d):
                if mod.stride == (2, 2):
                    mod.stride = (1, 1)
                if mod.kernel_size == (3, 3):
                    mod.dilation, mod.padding = (2, 2), (2, 2)
        for mod in self.body[5].modules():
            if isinstance(mod, nn.Conv2d) and mod.kernel_size == (1, 1):
                mod.stride = (1, 1)
        assert out_channels == 128, "layer2 of ResNet-18 emits 128 channels"
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.body(x), p=2.0, dim=1)


@torch.no_grad()
def calibrate_alignment(encoder: "SiameseEncoder", size: int = 256) -> float:
    from src.localizer.geometry import STRIDE

    was_training = encoder.training
    encoder.eval()
    centre = size // 2

    base = torch.zeros(1, 1, size, size, device=next(encoder.parameters()).device)
    impulse = base.clone()
    impulse[0, 0, centre, centre] = 1.0

    resp = (encoder.body(impulse) - encoder.body(base)).abs().sum(dim=1)[0]

    weights = resp / resp.sum().clamp_min(1e-12)
    idx = torch.arange(resp.shape[-1], dtype=torch.float32, device=resp.device)
    cx = float((weights.sum(dim=0) * idx).sum())
    cy = float((weights.sum(dim=1) * idx).sum())

    if was_training:
        encoder.train()
    return float(centre - STRIDE * (cx + cy) / 2.0)
