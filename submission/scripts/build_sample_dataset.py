#!/usr/bin/env python3
"""Rebuild results/sample_dataset/ -- the committed worked example of the
full generate -> predict pipeline.

The sample dataset spans all four {imaging_noise_profile} x
{geometric_profile} conditions, which generate_dataset.py cannot do in one
invocation (each run writes a single condition). This script drives the real
generate_dataset.py entry point once per condition, from a distinct seed
block so no canvas is shared between conditions, then merges the four
outputs into one flat directory with contiguous ids and a single manifest.

    python scripts/build_sample_dataset.py --out-dir results/sample_dataset

Manifest paths are written relative to the submission/ root, so the
committed CSV stays valid on any machine. Follow with:

    python localize.py --manifest results/sample_dataset/manifest.csv \
        --output results/sample_dataset/predictions.csv
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# One seed block per condition, spaced so the canvases never overlap. All
# lie inside LocalizerConfig's test split range [200000, 200500).
CONDITIONS = [
    {"imaging_noise_profile": "normal", "geometric_profile": "normal", "seed": 200000},
    {"imaging_noise_profile": "harsh", "geometric_profile": "normal", "seed": 200100},
    {"imaging_noise_profile": "normal", "geometric_profile": "drift", "seed": 200200},
    {"imaging_noise_profile": "harsh", "geometric_profile": "drift", "seed": 200300},
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="results/sample_dataset",
                   help="rebuilt in place; existing reference/ and search/ "
                        "image directories are removed first")
    p.add_argument("--per-condition", type=int, default=8,
                   help="pairs per condition (default 8 -> 32 total)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(REPO_ROOT, args.out_dir)
    ref_dir = os.path.join(out_dir, "reference")
    search_dir = os.path.join(out_dir, "search")
    for d in (ref_dir, search_dir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    merged, fieldnames = [], None
    with tempfile.TemporaryDirectory() as tmp:
        for condition in CONDITIONS:
            stage = os.path.join(tmp, f"{condition['imaging_noise_profile']}_"
                                       f"{condition['geometric_profile']}")
            cmd = [sys.executable, "generate_dataset.py",
                   "--split", "test",
                   "--num-samples", str(args.per_condition),
                   "--seed", str(condition["seed"]),
                   "--output-dir", stage,
                   "--imaging-noise-profile", condition["imaging_noise_profile"],
                   "--geometric-profile", condition["geometric_profile"]]
            print("+ " + " ".join(cmd))
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)

            with open(os.path.join(stage, "manifest.csv"), newline="") as f:
                rows = list(csv.DictReader(f))
            fieldnames = fieldnames or list(rows[0].keys())
            for row in rows:
                new_id = len(merged)
                name = f"{new_id:05d}.png"
                shutil.copyfile(row["reference_path"], os.path.join(ref_dir, name))
                shutil.copyfile(row["search_path"], os.path.join(search_dir, name))
                row["id"] = new_id
                # Relative to the submission/ root, matching every other
                # path in this repo -- an absolute staging path would not
                # survive being committed.
                row["reference_path"] = os.path.join(args.out_dir, "reference", name)
                row["search_path"] = os.path.join(args.out_dir, "search", name)
                merged.append(row)

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    print(f"\nwrote {len(merged)} pairs across {len(CONDITIONS)} conditions "
          f"to {args.out_dir}")
    print("next: python localize.py --manifest "
          f"{os.path.join(args.out_dir, 'manifest.csv')} "
          f"--output {os.path.join(args.out_dir, 'predictions.csv')}")


if __name__ == "__main__":
    main()
