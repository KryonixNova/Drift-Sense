from __future__ import annotations

import torch
import torch.nn as nn

from src.localizer.config import LocalizerConfig
from src.localizer.context_head import ContextHead
from src.localizer.correlation import dense_correlation
from src.localizer.decode import decode
from src.localizer.encoder import SiameseEncoder, calibrate_alignment


class DriftSenseLocalizer(nn.Module):
    def __init__(self, config: LocalizerConfig | None = None,
                 use_context: bool = True):
        super().__init__()
        self.config = config or LocalizerConfig()
        self.use_context = use_context
        self.encoder = SiameseEncoder(out_channels=self.config.corr_channels)
        self.context = (
            ContextHead(in_channels=self.config.corr_channels,
                        mid_channels=self.config.context_channels,
                        dilations=self.config.context_dilations)
            if use_context else None
        )
        self.b2_scale = nn.Parameter(torch.tensor(1.0 / self.config.corr_channels))
        self.b2_bias = nn.Parameter(torch.tensor(-4.0))
        self.register_buffer("align_offset", torch.zeros(()))

    def calibrate(self) -> float:
        off = calibrate_alignment(self.encoder)
        self.align_offset.fill_(off)
        return off

    def forward(self, reference: torch.Tensor, search: torch.Tensor):
        corr = dense_correlation(self.encoder(search), self.encoder(reference))
        if self.context is not None:
            return self.context(corr)
        heatmap = corr.sum(dim=1) * self.b2_scale + self.b2_bias
        offset = torch.zeros(corr.shape[0], 2, *heatmap.shape[-2:],
                             device=corr.device, dtype=corr.dtype)
        return heatmap, offset

    @torch.no_grad()
    def predict(self, reference: torch.Tensor, search: torch.Tensor) -> dict:
        logits, offset = self(reference, search)
        heatmap = torch.sigmoid(logits)
        return decode(heatmap, offset,
                      peak_tie_ratio=self.config.peak_tie_ratio,
                      nms_kernel=self.config.nms_kernel,
                      align_offset=float(self.align_offset))
