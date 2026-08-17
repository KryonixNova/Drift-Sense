from __future__ import annotations

import torch
import torch.nn.functional as F

from src.localizer.geometry import CORR, HALF, STRIDE


def decode(heatmap: torch.Tensor, offset: torch.Tensor,
           peak_tie_ratio: float = 0.95, nms_kernel: int = 3,
           align_offset: float = 0.0) -> dict:
    b = heatmap.shape[0]
    dev = heatmap.device

    pooled = F.max_pool2d(heatmap.unsqueeze(1), nms_kernel,
                          stride=1, padding=nms_kernel // 2).squeeze(1)
    peaks = heatmap * (heatmap >= pooled - 1e-9).float()

    flat = peaks.reshape(b, -1)
    best = flat.max(dim=1).values

    grid = torch.arange(CORR, dtype=torch.float32, device=dev)
    mid = (CORR - 1) / 2.0
    dist = ((grid.view(1, CORR, 1) - mid) ** 2 +
            (grid.view(1, 1, CORR) - mid) ** 2).sqrt().expand(b, CORR, CORR)

    tied = peaks >= (best.view(b, 1, 1) * peak_tie_ratio)
    tied &= peaks > 0
    masked = torch.where(tied, dist, torch.full_like(dist, float("inf")))
    masked_flat = masked.reshape(b, -1)
    min_dist = masked_flat.min(dim=1, keepdim=True).values
    dist_candidates = masked_flat <= (min_dist + 1e-4)
    peaks_flat = peaks.reshape(b, -1)
    tie_break_peaks = torch.where(
        dist_candidates, peaks_flat, torch.full_like(peaks_flat, float("-inf")))
    chosen = tie_break_peaks.argmax(dim=1)
    row, col = chosen // CORR, chosen % CORR

    idx = torch.arange(b, device=dev)
    dx = offset[idx, 0, row, col]
    dy = offset[idx, 1, row, col]

    x = STRIDE * col.float() + HALF + align_offset + dx
    y = STRIDE * row.float() + HALF + align_offset + dy

    excl = ((grid.view(1, CORR, 1) - row.view(b, 1, 1).float()) ** 2 +
            (grid.view(1, 1, CORR) - col.view(b, 1, 1).float()) ** 2) > 9.0
    runner_up = torch.where(excl, peaks, torch.zeros_like(peaks))
    runner_up = runner_up.reshape(b, -1).max(dim=1).values
    confidence = peaks[idx, row, col] - runner_up

    return {"x": x, "y": y, "confidence": confidence}
