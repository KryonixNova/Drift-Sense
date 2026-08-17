from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_heatmap_loss(pred_logits: torch.Tensor, target: torch.Tensor,
                       alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    pred_logits = pred_logits.float()
    target = target.float()
    p = torch.sigmoid(pred_logits).clamp(1e-6, 1 - 1e-6)
    pos = target.ge(1.0 - 1e-6).float()
    neg = 1.0 - pos

    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * neg

    n_pos = pos.sum().clamp_min(1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def offset_loss(pred: torch.Tensor, target: torch.Tensor,
                peak_cell: torch.Tensor) -> torch.Tensor:
    b = pred.shape[0]
    idx = torch.arange(b, device=pred.device)
    col, row = peak_cell[:, 0], peak_cell[:, 1]
    p = pred[idx, :, row, col]
    t = target[idx, :, row, col]
    return F.l1_loss(p, t)


def hard_negative_loss(pred_logits: torch.Tensor, peak_cell: torch.Tensor,
                       radius_cells: int = 12, margin: float = 1.0) -> torch.Tensor:
    from src.localizer.geometry import CORR

    b = pred_logits.shape[0]
    dev = pred_logits.device
    grid = torch.arange(CORR, dtype=torch.float32, device=dev)
    col = peak_cell[:, 0].view(b, 1, 1).float()
    row = peak_cell[:, 1].view(b, 1, 1).float()

    far = ((grid.view(1, CORR, 1) - row) ** 2 +
           (grid.view(1, 1, CORR) - col) ** 2) > float(radius_cells) ** 2

    neg_inf = torch.finfo(pred_logits.dtype).min
    rival = torch.where(far, pred_logits, torch.full_like(pred_logits, neg_inf))
    rival = rival.reshape(b, -1).max(dim=1).values

    idx = torch.arange(b, device=dev)
    true_score = pred_logits[idx, peak_cell[:, 1], peak_cell[:, 0]]
    return F.relu(rival - true_score + margin).mean()
