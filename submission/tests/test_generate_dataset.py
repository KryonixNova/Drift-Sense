import csv
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_generate_dataset_writes_pngs_and_manifest(tmp_path):
    out_dir = tmp_path / "out"
    cmd = [sys.executable, "generate_dataset.py", "--num-samples", "3",
           "--split", "test", "--output-dir", str(out_dir),
           "--geometric-profile", "drift"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr

    manifest_path = out_dir / "manifest.csv"
    assert manifest_path.exists()
    rows = list(csv.DictReader(open(manifest_path)))
    assert len(rows) == 3
    expected_cols = {"id", "architecture", "reference_path", "search_path", "gt_x", "gt_y",
                     "canvas_seed", "crop_index", "crop_x0_fine", "crop_y0_fine",
                     "jitter_profile", "imaging_noise_profile", "geometric_profile",
                     "scale_ratio", "rotation_deg",
                     "mats_m", "mats_n", "strip_width_nm", "linewidth_bias_nm",
                     "corner_rounding_px",
                     "search_spot_size_nm", "search_dose", "search_shear_amplitude_px",
                     "search_drift_jitter_px", "search_detector_noise_sigma",
                     "search_astigmatism_ratio", "search_vignette_strength",
                     "search_barrel_distortion_k", "search_charging_streak_prob",
                     "search_charging_streak_intensity", "search_speckle_sigma",
                     "search_salt_pepper_prob",
                     "ref_spot_size_nm", "ref_dose", "ref_detector_noise_sigma",
                     "ref_drift_jitter_px", "ref_astigmatism_ratio",
                     "ref_vignette_strength", "ref_barrel_distortion_k",
                     "ref_charging_streak_prob", "ref_charging_streak_intensity",
                     "ref_speckle_sigma", "ref_salt_pepper_prob"}
    assert set(rows[0]) == expected_cols
    for row in rows:
        assert row["architecture"] == "dram"
        assert row["geometric_profile"] == "drift"
        ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
        assert ref_img.shape == (1000, 1000)
        assert search_img.shape == (1000, 1000)
        # The spec's reference is a 100x *high-resolution* capture. An image
        # upscaled from the model's 100x100 working view would have no
        # structure at the 1 nm/px scale, so its finest detail would sit
        # far below this floor (measured ~0.9 for the upscaled form).
        detail = float(abs(cv2.Laplacian(ref_img.astype("float32"), cv2.CV_32F)).mean())
        assert detail > 2.0, f"reference {row['reference_path']} is not a native 100x capture"
        # Recorded noise settings must be the concrete per-pair values.
        assert 0.0 < float(row["ref_dose"]) and 0.0 < float(row["search_dose"])
        assert 0.0 < float(row["search_spot_size_nm"])
        assert 0.9 <= float(row["scale_ratio"]) <= 1.1
        assert abs(float(row["rotation_deg"])) <= 2.0


def test_generate_dataset_rejects_seed_outside_split_range(tmp_path):
    out_dir = tmp_path / "out"
    cmd = [sys.executable, "generate_dataset.py", "--num-samples", "1",
           "--split", "test", "--seed", "0", "--output-dir", str(out_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
    assert "outside" in (result.stdout + result.stderr)
