import pytest
import torch

from src.localizer.context_head import (
    ContextHead, effective_receptive_field, theoretical_rf,
)
from src.localizer.geometry import CORR


def test_theoretical_rf_matches_the_hand_computation():
    assert theoretical_rf((1, 2, 4, 8, 16, 32, 64)) == 255
    assert theoretical_rf((1, 2, 4, 8, 16, 32, 64, 1, 1)) == 259


def test_default_rf_covers_the_whole_correlation_map():
    assert theoretical_rf((1, 2, 4, 8, 16, 32, 64, 1, 1)) >= CORR


def test_output_shapes():
    head = ContextHead().eval()
    with torch.no_grad():
        hm, off = head(torch.randn(2, 128, CORR, CORR))
    assert hm.shape == (2, CORR, CORR)
    assert off.shape == (2, 2, CORR, CORR)


def test_resolution_is_preserved_throughout():
    head = ContextHead().eval()
    with torch.no_grad():
        hm, _ = head(torch.randn(1, 128, 64, 64))
    assert hm.shape == (1, 64, 64)


@pytest.mark.slow
def test_effective_receptive_field_meets_the_gate():
    head = ContextHead().eval()
    erf = effective_receptive_field(head, size=CORR)
    assert erf >= 113.0, (
        f"effective RF {erf:.1f} < 113 cells. Per the spec, add the "
        f"conditional axial-attention block after the dilated stack."
    )
    assert erf < CORR, "saturated: probe is degenerate, not a real measurement"


def test_content_free_input_does_not_create_a_border_or_corner_bias():
    torch.manual_seed(0)
    head = ContextHead().eval()
    x = torch.full((1, 128, CORR, CORR), 0.5)
    with torch.no_grad():
        hm, _ = head(x)
    hm = hm[0]
    assert hm.std().item() < 1e-3, (
        f"heatmap std {hm.std().item():.6f} for content-free uniform input "
        f"is not ~0 -- border cells are statistically distinguishable from "
        f"interior cells purely from padding, reproducing the corner-"
        f"collapse bug seen in real M2 training."
    )
