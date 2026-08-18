import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_validation_report_produces_required_outputs(tmp_path, tiny_checkpoint):
    ckpt_path, _device = tiny_checkpoint
    out_dir = tmp_path / "results"
    cmd = [sys.executable, "scripts/validation_report.py",
           "--checkpoint", ckpt_path, "--n-per-condition", "3",
           "--out-dir", str(out_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr

    report = json.load(open(out_dir / "validation_report.json"))
    assert set(report["conditions"]) == {
        "noise=normal_geom=normal", "noise=harsh_geom=normal",
        "noise=normal_geom=drift", "noise=harsh_geom=drift",
    }
    for cond in report["conditions"].values():
        assert cond["n"] == 3
        # 0.5px is the spec's "sub-pixel performance where supported" column.
        for t in ("5.0", "4.0", "2.0", "1.0", "0.5"):
            assert f"pass_rate@{t}px" in cond
        assert "mean_error_px" in cond and "median_error_px" in cond
        assert "worst_error_px" in cond
        assert "runtime_ms_mean" in cond and "runtime_ms_median" in cond

    # The spec asks for results across target positions, scales and rotations,
    # not only across noise levels.
    strat = report["stratified"]
    assert set(strat) == {"by_target_position", "by_scale_ratio", "by_rotation",
                          "by_barrel_distortion"}
    for section in strat.values():
        assert section["note"]
        assert section["buckets"], "stratification produced no populated buckets"
        for bucket in section["buckets"].values():
            assert bucket["n"] >= 1
            assert "mean_error_px" in bucket and "worst_error_px" in bucket
            assert "pass_rate@0.5px" in bucket
    # Scale/rotation slices cover the drift half only; positions cover all.
    for key in ("by_target_position", "by_barrel_distortion"):
        assert sum(b["n"] for b in strat[key]["buckets"].values()) == 12
    for key in ("by_scale_ratio", "by_rotation"):
        assert sum(b["n"] for b in strat[key]["buckets"].values()) == 6

    assert "hardware" in report
    assert "python_version" in report
    assert "timing_method" in report
    fc = report["failure_case"]
    assert fc["root_cause"]
    # The root cause must cite the measured drivers, not guess from confidence.
    assert "border" in fc["root_cause"] and "barrel" in fc["root_cause"]
    assert "border_distance_px" in fc and "barrel_distortion_k" in fc

    md = (out_dir / "validation_report.md").read_text()
    assert "pass@0.5px" in md
    for heading in ("By target position", "By barrel distortion",
                    "By reference scale ratio", "By reference rotation"):
        assert heading in md
    assert (out_dir / "failure_case.png").stat().st_size > 0


def test_border_distance_measures_nearest_edge():
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.validation_report import border_distance, stratify

    assert border_distance(500.0, 500.0) == 500.0   # dead centre
    assert border_distance(60.0, 500.0) == 60.0     # near left edge
    assert border_distance(500.0, 940.0) == 60.0    # near bottom edge
    assert border_distance(940.0, 60.0) == 60.0     # nearest of the two

    from scripts.validation_report import (
        BARREL_BUCKETS, POSITION_BUCKETS, ROTATION_BUCKETS, SCALE_BUCKETS)
    for buckets in (POSITION_BUCKETS, SCALE_BUCKETS, ROTATION_BUCKETS,
                    BARREL_BUCKETS):
        for label, _lo, _hi in buckets:
            assert "|" not in label, (
                f"bucket label {label!r} contains a pipe, which would split "
                f"the Markdown table cell it is rendered into")

    samples = [{"error": 1.0, "d": 10.0}, {"error": 3.0, "d": 20.0},
               {"error": 9.0, "d": 200.0}]
    out = stratify(samples, "d", [("near", 0.0, 100.0), ("far", 100.0, 1e9),
                                  ("empty", 1e9, 2e9)])
    assert "empty" not in out, "empty buckets must be dropped, not reported as n=0"
    assert out["near"]["n"] == 2 and out["near"]["mean_error_px"] == 2.0
    assert out["far"]["n"] == 1 and out["far"]["worst_error_px"] == 9.0
    assert out["near"]["pass_rate@2.0px"] == 0.5
