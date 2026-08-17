import cv2
import numpy as np
import pytest

from src.pipeline import GenerationParams, generate_sample
from src.presets import PRESETS

MAX_MEAN_ABS_DIFF = 30.0


@pytest.mark.parametrize("architecture", list(PRESETS.keys()))
def test_ground_truth_patch_matches_reference(architecture):
    rng = np.random.default_rng(0)
    params = GenerationParams(
        beam_spot_size_nm=1.0,
        dose_reference=1e6,
        dose_search=1e6,
        shear_amplitude_px=0.0,
        drift_jitter_px=0.0,
        detector_noise_sigma_ref=0.0,
        detector_noise_sigma_search=0.0,
    )
    sample = generate_sample(architecture, rng, params)

    x0, y0, w, h = (int(round(v)) for v in sample["gt_box"])
    patch_at_gt = sample["search_img"][y0:y0 + h, x0:x0 + w]
    reference_downsampled = cv2.resize(sample["reference_img"], (w, h), interpolation=cv2.INTER_AREA)

    mean_abs_diff = np.abs(patch_at_gt.astype(int) - reference_downsampled.astype(int)).mean()
    assert mean_abs_diff < MAX_MEAN_ABS_DIFF


def test_image_shapes_and_ground_truth_in_bounds():
    rng = np.random.default_rng(1)
    params = GenerationParams()
    sample = generate_sample("dram_1x", rng, params)

    assert sample["reference_img"].shape == (1000, 1000)
    assert sample["search_img"].shape == (1000, 1000)
    assert 0 <= sample["gt_x"] <= 1000
    assert 0 <= sample["gt_y"] <= 1000
