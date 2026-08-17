#!/usr/bin/env python3
"""CLI to generate a DRAM localization dataset split: PNG reference/search
image pairs plus a manifest.csv of ground truth and generation params.

Example:
    python generate_dataset.py --num-samples 20 --split train \
        --presets dram_1x dram_dense --output-dir ./output --seed 42
"""

import argparse
import csv
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

from dram_localization_pipeline import DRAM_PRESETS, generate_sample

REFERENCE_ZOOM_RATIO = 10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--presets", nargs="+", default=list(DRAM_PRESETS.keys()),
                   choices=list(DRAM_PRESETS.keys()))
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--boundary-bias", type=float, default=0.0)
    p.add_argument("--die-block-rows", type=int, default=1)
    p.add_argument("--die-block-cols", type=int, default=1)
    p.add_argument("--die-street-width-nm", type=float, default=100.0)
    p.add_argument("--workers", type=int, default=1,
                   help="generate samples in parallel across this many "
                        "processes -- each sample is fully independent "
                        "(no shared state), so this is close to linear "
                        "speedup up to the machine's core count. Default 1 "
                        "reproduces the original sequential behavior.")
    return p.parse_args()


def _reference_search_bbox_px(sample):
    ref_h_px, ref_w_px = sample.reference_img.shape[:2]
    w = ref_w_px / sample.zoom_ratio
    h = ref_h_px / sample.zoom_ratio
    cx, cy = sample.true_center_px
    return (cx - w / 2, cy - h / 2, w, h)


def _sample_params_kw(rng, presets, die_block_rows, die_block_cols,
                       boundary_bias, die_street_width_nm):
    multi_block = die_block_rows * die_block_cols > 1
    kw = dict(
        die_block_rows=die_block_rows, die_block_cols=die_block_cols,
        die_street_width_nm=die_street_width_nm, boundary_bias=boundary_bias,
        zoom_ratio=REFERENCE_ZOOM_RATIO,
    )
    if multi_block:
        kw["block_presets"] = list(presets)
        preset_label = "+".join(sorted(presets))
    else:
        preset_name = presets[int(rng.integers(0, len(presets)))]
        kw.update(DRAM_PRESETS[preset_name])
        preset_label = preset_name
    return kw, preset_label


def _render_one(i, seed, params_kw, preset_label, tmp_dir, ref_dir, search_dir,
                 die_block_rows, die_block_cols, boundary_bias):
    sample = generate_sample(seed=seed, tmp_dir=tmp_dir, **params_kw)

    ref_path = os.path.join(ref_dir, f"{i:05d}.png")
    search_path = os.path.join(search_dir, f"{i:05d}.png")
    cv2.imwrite(ref_path, sample.reference_img)
    cv2.imwrite(search_path, sample.search_img)

    if sample.defect_bbox_px is not None:
        bx0, by0, bx1, by1 = sample.defect_bbox_px
        gt_box = (bx0, by0, bx1 - bx0, by1 - by0)
    else:
        gt_box = _reference_search_bbox_px(sample)

    gx0, gy0, gw, gh = gt_box
    row = {
        "id": i,
        "reference_path": ref_path,
        "search_path": search_path,
        "gt_x": sample.true_center_px[0],
        "gt_y": sample.true_center_px[1],
        "gt_box_x": gx0, "gt_box_y": gy0, "gt_box_w": gw, "gt_box_h": gh,
        "preset": preset_label,
        "zoom_ratio": sample.zoom_ratio,
        "die_block_rows": die_block_rows,
        "die_block_cols": die_block_cols,
        "boundary_bias": boundary_bias,
        "seed": sample.seed,
    }
    return i, preset_label, row


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    split_dir = os.path.join(args.output_dir, args.split)
    ref_dir = os.path.join(split_dir, "reference")
    search_dir = os.path.join(split_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    manifest_path = os.path.join(split_dir, "manifest.csv")
    fieldnames = [
        "id", "reference_path", "search_path", "gt_x", "gt_y",
        "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h", "preset",
        "zoom_ratio", "die_block_rows", "die_block_cols", "boundary_bias", "seed",
    ]

    jobs = []
    for i in range(args.num_samples):
        params_kw, preset_label = _sample_params_kw(
            rng, args.presets, args.die_block_rows, args.die_block_cols,
            args.boundary_bias, args.die_street_width_nm)
        jobs.append((i, args.seed + i, params_kw, preset_label))

    with tempfile.TemporaryDirectory() as tmp_dir, \
            open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        rows_by_id = {}
        if args.workers <= 1:
            for i, seed, params_kw, preset_label in jobs:
                i, preset_label, row = _render_one(
                    i, seed, params_kw, preset_label, tmp_dir, ref_dir, search_dir,
                    args.die_block_rows, args.die_block_cols, args.boundary_bias)
                rows_by_id[i] = row
                print(f"[{i + 1}/{args.num_samples}] {preset_label} -> "
                      f"gt=({row['gt_x']:.1f}, {row['gt_y']:.1f})")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(_render_one, i, seed, params_kw, preset_label,
                                tmp_dir, ref_dir, search_dir, args.die_block_rows,
                                args.die_block_cols, args.boundary_bias)
                    for i, seed, params_kw, preset_label in jobs
                ]
                done = 0
                for fut in as_completed(futures):
                    i, preset_label, row = fut.result()
                    rows_by_id[i] = row
                    done += 1
                    print(f"[{done}/{args.num_samples}] {preset_label} -> "
                          f"gt=({row['gt_x']:.1f}, {row['gt_y']:.1f})")

        for i in range(args.num_samples):
            writer.writerow(rows_by_id[i])

    print(f"Wrote {args.num_samples} samples to {split_dir}")


if __name__ == "__main__":
    main()
