import torch

from src.localizer.config import LocalizerConfig
from src.localizer.geometry import CORR
from src.localizer.model import DriftSenseLocalizer


def _inputs(b=1):
    return torch.randn(b, 1, 100, 100), torch.randn(b, 1, 1000, 1000)


def test_forward_shapes_with_context():
    m = DriftSenseLocalizer(LocalizerConfig()).eval()
    with torch.no_grad():
        hm, off = m(*_inputs())
    assert hm.shape == (1, CORR, CORR)
    assert off.shape == (1, 2, CORR, CORR)


def test_forward_shapes_without_context_b2_ablation():
    m = DriftSenseLocalizer(LocalizerConfig(), use_context=False).eval()
    with torch.no_grad():
        hm, off = m(*_inputs())
    assert hm.shape == (1, CORR, CORR)
    assert off.shape == (1, 2, CORR, CORR)


def test_b2_ablation_has_no_context_parameters():
    with_ctx = DriftSenseLocalizer(LocalizerConfig())
    without = DriftSenseLocalizer(LocalizerConfig(), use_context=False)
    assert sum(p.numel() for p in without.parameters()) < \
           sum(p.numel() for p in with_ctx.parameters())


def test_predict_returns_only_the_three_contract_outputs():
    m = DriftSenseLocalizer(LocalizerConfig()).eval()
    with torch.no_grad():
        out = m.predict(*_inputs())
    assert set(out) == {"x", "y", "confidence"}
    assert "localization_error" not in out


def test_predictions_lie_inside_the_valid_range():
    m = DriftSenseLocalizer(LocalizerConfig()).eval()
    with torch.no_grad():
        out = m.predict(*_inputs(b=2))
    assert torch.all(out["x"] >= 48.0) and torch.all(out["x"] <= 952.0)
    assert torch.all(out["y"] >= 48.0) and torch.all(out["y"] <= 952.0)


def test_gradients_reach_the_encoder():
    m = DriftSenseLocalizer(LocalizerConfig())
    hm, _ = m(*_inputs())
    hm.sum().backward()
    grads = [p.grad for p in m.encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_b2_ablation_heatmap_logits_do_not_saturate_sigmoid():
    m = DriftSenseLocalizer(LocalizerConfig(), use_context=False).eval()
    with torch.no_grad():
        hm, _ = m(*_inputs())
    probs = torch.sigmoid(hm)
    frac_unsaturated = ((probs > 0.01) & (probs < 0.99)).float().mean()
    assert frac_unsaturated > 0.9


def test_gradients_reach_the_encoder_without_context_b2_ablation():
    m = DriftSenseLocalizer(LocalizerConfig(), use_context=False)
    hm, _ = m(*_inputs())
    torch.sigmoid(hm).sum().backward()
    grads = [p.grad for p in m.encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_b2_predict_does_not_collapse_once_bias_goes_negative():
    torch.manual_seed(0)
    m = DriftSenseLocalizer(LocalizerConfig(), use_context=False).eval()
    with torch.no_grad():
        m.b2_bias.fill_(-6.0)
        ref, search = _inputs(b=3)
        out = m.predict(ref, search)

    xs, ys = out["x"], out["y"]
    all_identical = all(
        torch.allclose(xs[i], xs[0]) and torch.allclose(ys[i], ys[0])
        for i in range(1, xs.shape[0])
    )
    assert not all_identical
