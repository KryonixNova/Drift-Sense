from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 1, 1)


def theoretical_rf(dilations, kernel: int = 3) -> int:
    return 1 + sum((kernel - 1) * d for d in dilations)


class _DilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=dilation,
                              dilation=dilation, padding_mode="replicate",
                              bias=False)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))


class ContextHead(nn.Module):
    def __init__(self, in_channels: int = 128, mid_channels: int = 64,
                 dilations=DEFAULT_DILATIONS):
        super().__init__()
        self.dilations = tuple(dilations)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[_DilatedBlock(mid_channels, d) for d in self.dilations]
        )
        self.heatmap_head = nn.Conv2d(mid_channels, 1, 1)
        self.offset_head = nn.Conv2d(mid_channels, 2, 1)
        nn.init.constant_(self.heatmap_head.bias, -4.0)

    @property
    def receptive_field(self) -> int:
        return theoretical_rf(self.dilations)

    def forward(self, corr: torch.Tensor):
        f = self.blocks(self.reduce(corr))
        return self.heatmap_head(f).squeeze(1), self.offset_head(f)


def effective_receptive_field(head: "ContextHead", size: int = 226) -> float:
    head = head.eval()
    torch.manual_seed(0)
    x = torch.randn(1, head.reduce[0].in_channels, size, size) * 0.01
    x.requires_grad_(True)
    hm, _ = head(x)
    hm[0, size // 2, size // 2].backward()

    g = x.grad.abs().sum(dim=(0, 1))
    total = g.sum().clamp_min(1e-12)
    mid = size // 2
    for half in range(1, size // 2 + 1):
        window = g[max(mid - half, 0):mid + half + 1,
                   max(mid - half, 0):mid + half + 1]
        if float(window.sum() / total) >= 0.95:
            return float(2 * half + 1)
    return float(size)
