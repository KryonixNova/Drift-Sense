import torch

from src.localizer.geometry import CORR
from src.localizer.losses import focal_heatmap_loss, offset_loss
from src.localizer.targets import build_targets


def test_focal_loss_is_near_zero_for_a_perfect_prediction():
    t = build_targets(torch.tensor([500.0]), torch.tensor([500.0]))
    logits = torch.logit(t["heatmap"].clamp(1e-4, 1 - 1e-4))
    assert float(focal_heatmap_loss(logits, t["heatmap"])) < 0.4


def test_focal_loss_penalises_a_confident_wrong_peak():
    t = build_targets(torch.tensor([500.0]), torch.tensor([500.0]))
    wrong = torch.full((1, CORR, CORR), -8.0)
    wrong[0, 10, 10] = 8.0
    right = torch.logit(t["heatmap"].clamp(1e-4, 1 - 1e-4))
    assert float(focal_heatmap_loss(wrong, t["heatmap"])) > \
           float(focal_heatmap_loss(right, t["heatmap"]))


def test_focal_loss_is_differentiable():
    t = build_targets(torch.tensor([300.0]), torch.tensor([300.0]))
    logits = torch.zeros(1, CORR, CORR, requires_grad=True)
    focal_heatmap_loss(logits, t["heatmap"]).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_focal_loss_is_finite_under_fp16_amp_saturated_logits():
    t = build_targets(torch.tensor([500.0]), torch.tensor([500.0]))
    logits_fp16 = torch.full((1, CORR, CORR), 15.0, dtype=torch.float16)
    loss = focal_heatmap_loss(logits_fp16, t["heatmap"])
    assert torch.isfinite(loss).all()


def test_offset_loss_zero_when_exact():
    t = build_targets(torch.tensor([437.3]), torch.tensor([612.9]))
    assert float(offset_loss(t["offset"], t["offset"], t["peak_cell"])) == 0.0


def test_offset_loss_only_scores_the_ground_truth_cell():
    t = build_targets(torch.tensor([437.3]), torch.tensor([612.9]))
    polluted = t["offset"].clone()
    polluted[0, :, 0, 0] += 99.0
    assert float(offset_loss(polluted, t["offset"], t["peak_cell"])) == 0.0


def test_offset_loss_grows_with_error():
    t = build_targets(torch.tensor([437.3]), torch.tensor([612.9]))
    col, row = t["peak_cell"][0].tolist()
    off = t["offset"].clone()
    off[0, 0, row, col] += 1.5
    assert float(offset_loss(off, t["offset"], t["peak_cell"])) > 0.0


from src.localizer.losses import hard_negative_loss


def _peak_at(col, row):
    return torch.tensor([[col, row]])


def test_no_penalty_when_only_the_true_peak_is_strong():
    logits = torch.full((1, CORR, CORR), -8.0)
    logits[0, 100, 100] = 8.0
    assert float(hard_negative_loss(logits, _peak_at(100, 100))) == 0.0


def test_penalises_a_strong_distant_rival():
    logits = torch.full((1, CORR, CORR), -8.0)
    logits[0, 100, 100] = 4.0
    logits[0, 20, 20] = 8.0
    assert float(hard_negative_loss(logits, _peak_at(100, 100))) > 0.0


def test_ignores_rivals_inside_the_radius():
    logits = torch.full((1, CORR, CORR), -8.0)
    logits[0, 100, 100] = 4.0
    logits[0, 100, 103] = 8.0
    assert float(hard_negative_loss(logits, _peak_at(100, 100), radius_cells=12)) == 0.0


def test_radius_is_configurable():
    logits = torch.full((1, CORR, CORR), -8.0)
    logits[0, 100, 100] = 4.0
    logits[0, 100, 120] = 8.0
    inside = hard_negative_loss(logits, _peak_at(100, 100), radius_cells=24)
    outside = hard_negative_loss(logits, _peak_at(100, 100), radius_cells=6)
    assert float(inside) == 0.0
    assert float(outside) > 0.0


def test_hard_negative_loss_is_differentiable():
    logits = torch.full((1, CORR, CORR), -8.0, requires_grad=True).clone().detach()
    logits.requires_grad_(True)
    with torch.no_grad():
        logits[0, 20, 20] = 8.0
    hard_negative_loss(logits, _peak_at(100, 100)).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
