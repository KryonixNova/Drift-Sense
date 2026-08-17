import torch

from src.localizer.decode import decode
from src.localizer.geometry import CORR, cell_to_pixel
from src.localizer.targets import build_targets


def _blank():
    return torch.zeros(1, CORR, CORR), torch.zeros(1, 2, CORR, CORR)


def test_single_peak_decodes_to_its_pixel_centre():
    hm, off = _blank()
    hm[0, 70, 30] = 1.0
    out = decode(hm, off)
    assert float(out["x"][0]) == cell_to_pixel(30)
    assert float(out["y"][0]) == cell_to_pixel(70)


def test_offset_is_added_to_the_cell_centre():
    hm, off = _blank()
    hm[0, 70, 30] = 1.0
    off[0, 0, 70, 30] = 1.75
    off[0, 1, 70, 30] = -1.25
    out = decode(hm, off)
    assert float(out["x"][0]) == cell_to_pixel(30) + 1.75
    assert float(out["y"][0]) == cell_to_pixel(70) - 1.25


def test_tied_peaks_resolve_to_the_one_nearest_image_centre():
    hm, off = _blank()
    hm[0, 113, 113] = 1.0
    hm[0, 5, 5] = 1.0
    out = decode(hm, off)
    assert float(out["x"][0]) == cell_to_pixel(113)
    assert float(out["y"][0]) == cell_to_pixel(113)


def test_a_clearly_stronger_far_peak_still_wins():
    hm, off = _blank()
    hm[0, 113, 113] = 0.30
    hm[0, 5, 5] = 1.00
    out = decode(hm, off)
    assert float(out["x"][0]) == cell_to_pixel(5)


def test_confidence_is_the_margin_not_the_peak_height():
    hm_amb, off = _blank()
    hm_amb[0, 40, 40] = 0.9
    hm_amb[0, 150, 150] = 0.88

    hm_clear, _ = _blank()
    hm_clear[0, 40, 40] = 0.9
    hm_clear[0, 150, 150] = 0.05

    assert float(decode(hm_clear, off)["confidence"][0]) > \
           float(decode(hm_amb, off)["confidence"][0])


def test_decode_recovers_the_target_it_was_built_from():
    t = build_targets(torch.tensor([437.3]), torch.tensor([612.9]))
    out = decode(t["heatmap"], t["offset"])
    assert abs(float(out["x"][0]) - 437.3) < 1e-2
    assert abs(float(out["y"][0]) - 612.9) < 1e-2


def test_batch_is_handled():
    t = build_targets(torch.tensor([200.0, 800.0]), torch.tensor([300.0, 700.0]))
    out = decode(t["heatmap"], t["offset"])
    assert out["x"].shape == (2,)
    assert abs(float(out["x"][0]) - 200.0) < 1e-2
    assert abs(float(out["x"][1]) - 800.0) < 1e-2


def test_exact_distance_tie_resolves_to_the_higher_peak():
    hm, off = _blank()
    hm[0, 100, 100] = 0.99
    hm[0, 125, 125] = 1.00
    out = decode(hm, off)
    assert float(out["x"][0]) == cell_to_pixel(125)
    assert float(out["y"][0]) == cell_to_pixel(125)


def test_exact_distance_tie_does_not_default_to_first_index():
    hm, off = _blank()
    hm[0, 100, 100] = 1.00
    hm[0, 125, 125] = 0.99
    out = decode(hm, off)
    assert float(out["x"][0]) == cell_to_pixel(100)
    assert float(out["y"][0]) == cell_to_pixel(100)


def test_confidence_can_go_negative_when_tie_break_overrules_raw_score():
    hm, off = _blank()
    hm[0, 40, 40] = 0.9
    hm[0, 150, 150] = 0.88
    out = decode(hm, off)
    assert float(out["x"][0]) == cell_to_pixel(150)
    assert float(out["y"][0]) == cell_to_pixel(150)
    assert float(out["confidence"][0]) < 0
