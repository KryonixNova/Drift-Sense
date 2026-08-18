#!/usr/bin/env python3
"""Generate a persisted Drift-Sense localization dataset split: reference/
search PNG pairs plus a manifest.csv of ground truth and generation
metadata, using the exact same on-the-fly generator (src/localizer/data.py)
the training pipeline itself draws from -- so a dataset written by this
script is representative of what the model was actually trained/evaluated
on, not a separately-diverged generator.

Both images are written at 1000x1000 px, matching the spec's I/O contract,
and both are genuine captures at their own magnification: the search image
covers 10000x10000 nm at 10 nm/px (10x), and the reference covers
1000x1000 nm at 1 nm/px (100x). The reference is *not* the model's
10x-downsampled working view scaled back up -- it carries the full detail a
100x column resolves, which is what an evaluator's reference images will
also contain.

manifest.csv records one row per pair with the paths, ground-truth centre,
the random seed, and the concrete generation settings behind that pair --
pattern layout, every sampled search- and reference-side noise value, and
the scale/rotation applied to the reference.

Example:
    python generate_dataset.py --split test --num-samples 50 --seed 200000 \
        --output-dir ./output --imaging-noise-profile harsh \
        --geometric-profile drift
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from src.localizer.config import LocalizerConfig
from src.localizer.data import generate_canvas_bundle, sample_pair, split_seed_range
from src.localizer.geometry import REF_PX as REFERENCE_HIRES_PX

# Per-pair generation metadata mirrored into manifest.csv, so every pair can
# be reproduced and every reported result can be sliced by the exact
# acquisition settings that produced it (spec S4B: "Store the random seed,
# architecture, transformations, noise settings, scale, rotation and ground
# truth for every pair"). Column name = prefix + the sem_imaging keyword.
PATTERN_COLUMNS = ["mats_m", "mats_n", "strip_width_nm", "linewidth_bias_nm",
                   "corner_rounding_px"]
SEARCH_NOISE_COLUMNS = [
    "spot_size_nm", "dose", "shear_amplitude_px", "drift_jitter_px",
    "detector_noise_sigma", "astigmatism_ratio", "vignette_strength",
    "barrel_distortion_k", "charging_streak_prob", "charging_streak_intensity",
    "speckle_sigma", "salt_pepper_prob",
]
REFERENCE_NOISE_COLUMNS = [
    "spot_size_nm", "dose", "detector_noise_sigma", "drift_jitter_px",
    "astigmatism_ratio", "vignette_strength", "barrel_distortion_k",
    "charging_streak_prob", "charging_streak_intensity", "speckle_sigma",
    "salt_pepper_prob",
]


def _rounded(params: dict, keys: list, prefix: str) -> dict:
    return {f"{prefix}{k}": (params[k] if isinstance(params[k], int)
                             else round(float(params[k]), 5)) for k in keys}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, default=30)
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="which canvas-disjoint seed range to draw from")
    p.add_argument("--seed", type=int, default=None,
                   help="first canvas seed; defaults to the chosen split's "
                        "own range start. Must fall inside that range.")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--jitter-profile", default="normal",
                   choices=["normal", "zero", "shifted"])
    p.add_argument("--imaging-noise-profile", default="normal",
                   choices=["normal", "harsh"])
    p.add_argument("--geometric-profile", default="normal",
                   choices=["normal", "drift"])
    p.add_argument("--crops-per-canvas", type=int, default=1,
                   help="reference crops to draw per generated canvas "
                        "before moving to the next canvas seed")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = LocalizerConfig()
    lo, hi = split_seed_range(args.split, cfg)
    start_seed = args.seed if args.seed is not None else lo
    if not (lo <= start_seed < hi):
        raise SystemExit(
            f"--seed {start_seed} is outside the {args.split} split's own "
            f"canvas-disjoint range [{lo}, {hi}) -- pick a seed inside that "
            f"range so generated data doesn't overlap canvases used for a "
            f"different split.")

    ref_dir = os.path.join(args.output_dir, "reference")
    search_dir = os.path.join(args.output_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    fieldnames = (["id", "architecture", "reference_path", "search_path",
                   "gt_x", "gt_y", "canvas_seed", "crop_index",
                   "crop_x0_fine", "crop_y0_fine", "jitter_profile",
                   "imaging_noise_profile", "geometric_profile",
                   "scale_ratio", "rotation_deg"]
                  + PATTERN_COLUMNS
                  + [f"search_{k}" for k in SEARCH_NOISE_COLUMNS]
                  + [f"ref_{k}" for k in REFERENCE_NOISE_COLUMNS])

    i = 0
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        seed = start_seed
        while i < args.num_samples:
            if seed >= hi:
                raise SystemExit(
                    f"ran out of canvas seeds in the {args.split} split's "
                    f"range before reaching --num-samples {args.num_samples} "
                    f"(wrote {i}) -- lower --num-samples, raise "
                    f"--crops-per-canvas, or start from a lower --seed.")
            bundle = generate_canvas_bundle(seed, args.jitter_profile,
                                            args.imaging_noise_profile)
            for k in range(args.crops_per_canvas):
                if i >= args.num_samples:
                    break
                s = sample_pair(bundle, seed, k, args.imaging_noise_profile,
                                args.geometric_profile,
                                want_hires_reference=True)

                ref_path = os.path.join(ref_dir, f"{i:05d}.png")
                search_path = os.path.join(search_dir, f"{i:05d}.png")
                # The reference as the 100x column actually captured it:
                # REFERENCE_HIRES_PX px at 1 nm/px, not the model's
                # 10x-downsampled working view scaled back up.
                ref_hires = s["reference_img_hires_u8"]
                assert ref_hires.shape == (REFERENCE_HIRES_PX, REFERENCE_HIRES_PX)
                cv2.imwrite(ref_path, ref_hires)
                cv2.imwrite(search_path, bundle["search_img"])

                writer.writerow({
                    "id": i, "architecture": "dram",
                    "reference_path": ref_path, "search_path": search_path,
                    "gt_x": s["gt_x"], "gt_y": s["gt_y"], "canvas_seed": seed,
                    "crop_index": k,
                    "crop_x0_fine": s["crop_x0_fine"],
                    "crop_y0_fine": s["crop_y0_fine"],
                    "jitter_profile": args.jitter_profile,
                    "imaging_noise_profile": args.imaging_noise_profile,
                    "geometric_profile": args.geometric_profile,
                    "scale_ratio": round(s["scale_ratio"], 4),
                    "rotation_deg": round(s["rotation_deg"], 4),
                    **_rounded(bundle["pattern_params"], PATTERN_COLUMNS, ""),
                    **_rounded(bundle["search_noise_params"],
                               SEARCH_NOISE_COLUMNS, "search_"),
                    **_rounded(s["reference_noise_params"],
                               REFERENCE_NOISE_COLUMNS, "ref_"),
                })
                print(f"[{i + 1}/{args.num_samples}] seed={seed} crop={k} -> "
                      f"gt=({s['gt_x']:.1f}, {s['gt_y']:.1f})")
                i += 1
            seed += 1

    print(f"wrote {i} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
