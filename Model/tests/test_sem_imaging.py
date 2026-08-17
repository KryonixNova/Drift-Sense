import numpy as np

from src.sem_imaging import add_shot_noise, apply_barrel_distortion


def test_add_shot_noise_survives_nan_and_inf_input():
    img = np.full((8, 8), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    out = add_shot_noise(img, dose=np.nan, rng=rng)
    assert out.dtype == np.uint8
    assert np.isfinite(out.astype(np.float64)).all()


def test_add_shot_noise_survives_inf_and_implausibly_large_dose():
    img = np.full((8, 8), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    for dose in (np.inf, 1e300):
        out = add_shot_noise(img, dose=dose, rng=rng)
        assert out.dtype == np.uint8
        assert np.isfinite(out.astype(np.float64)).all()


def test_add_shot_noise_survives_zero_and_negative_dose():
    img = np.full((8, 8), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    for dose in (0.0, -5.0):
        out = add_shot_noise(img, dose=dose, rng=rng)
        assert out.dtype == np.uint8
        assert np.isfinite(out.astype(np.float64)).all()


def test_add_shot_noise_normal_case_unaffected():
    img = np.full((8, 8), 128, dtype=np.uint8)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    out_a = add_shot_noise(img, dose=400.0, rng=rng_a)
    out_b = add_shot_noise(img, dose=400.0, rng=rng_b)
    np.testing.assert_array_equal(out_a, out_b)


def test_apply_barrel_distortion_survives_extreme_k_without_crashing():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    for k in (1e10, -1e10, 1e20, 1e37, -1e37, 1e300):
        out = apply_barrel_distortion(img.copy(), k)
        assert out.dtype == np.uint8
        assert out.shape == img.shape
        assert np.isfinite(out.astype(np.float64)).all()


def test_apply_barrel_distortion_normal_case_nearly_unaffected():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    for k in (-0.08, -0.02, 0.02, 0.08):
        old = _old_float32_barrel_distortion(img.copy(), k)
        new = apply_barrel_distortion(img.copy(), k)
        assert new.dtype == np.uint8
        assert new.shape == img.shape
        max_diff = np.abs(old.astype(int) - new.astype(int)).max()
        assert max_diff <= 2, f"k={k}: max pixel diff {max_diff} exceeds precision tolerance"


def _old_float32_barrel_distortion(img, k):
    import cv2
    if k == 0.0:
        return img
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - cx) / cx
    ny = (yy - cy) / cy
    r2 = nx ** 2 + ny ** 2
    factor = 1.0 + k * r2
    map_x = (nx * factor) * cx + cx
    map_y = (ny * factor) * cy + cy
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
