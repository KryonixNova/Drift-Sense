<div align="center">

# Drift-Sense

### Synthetic SEM data generation + deep-learned reference localization for wafer inspection

**SEMI x IESA Hackathon 2026 — Applied Materials Problem Statement:
Drift-Sense — AI-Powered Navigation-Error Recovery for Wafer Inspection Tools**

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-ee4c2c)
![OpenCV](https://img.shields.io/badge/OpenCV-headless-5C3EE8)
![Hackathon](https://img.shields.io/badge/SEMI%20x%20IESA-Hackathon%202026-orange)

[**→ Graded submission**](submission/) · [Results](submission/results/validation_report.md) · [References](submission/references/README.md)

</div>

---

When a wafer inspection tool's stage drifts off its intended coordinates, it
needs to relocate a known reference feature inside a wider search image to
recover its true position — fast, and without a human in the loop.

Given a 1000x1000 reference image at 100x magnification and a 1000x1000
search image at 10x covering the same region, this project returns the
`(x, y)` centre of the reference inside the search image — on a repetitive
DRAM lattice where a naive best-match search finds many equally good
answers and picks the wrong one.

## Results

Measured on **200 independently generated pairs** — one fresh canvas per
pair — against the shipped `production_v3` checkpoint:

| | Median error | Pass @ 5px | Runtime / pair |
|---|---|---|---|
| Normal acquisition noise | **0.63–0.77 px** | **100%** | ~20 ms |
| Harsh acquisition noise | 1.35–1.37 px | 84% | ~20 ms |
| **Pooled (n=200)** | **0.99 px** | — | ~20 ms |

Runtime is a median over single pairs on an RTX 5060 Ti, Python 3.12.13,
timed with `time.perf_counter()` around the model call.

The error distribution is bimodal rather than spread: most pairs land
sub-pixel, while 5 of 200 (2.5%) miss by more than 50px, because on a
periodic lattice a wrong answer is a *different cell* — hundreds of pixels
away, not a few. Excluding those five, the mean over the remaining 195 is
1.76px. Read the median and pass rates, not the pooled mean.

Nearly all of the harsh-profile error traces to one generation setting:
barrel distortion of the search image. With no barrel warp the model clears
5px on **116 of 116** pairs. Full per-condition and stratified tables, the
failure case, and an important caveat about how ground truth interacts with
that warp are in
[`submission/results/validation_report.md`](submission/results/validation_report.md)
and §3/§12 of [`submission/README.md`](submission/README.md).

## Quick start

```bash
cd submission
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# localize one pair (prints "x,y,confidence")
python localize.py --reference ref.png --search search.png
```

Full setup notes, every command, the coordinate convention and the I/O
contract are in [`submission/README.md`](submission/README.md).

## Where the submission lives

**[`submission/`](submission/)** is the graded deliverable, laid out to
match the hackathon's recommended submission structure: a procedural
DRAM-SEM generator (`generate_dataset.py`) plus a deep-learned Siamese
reference-localization model (`localize.py`), training/eval scripts,
tests, results, and citations, all self-contained in that one folder. Start
there — [`submission/README.md`](submission/README.md) has the full write-up,
setup instructions, and results table.

> **`submission/solution_presentation.pptx` is not yet added.** The
> mandatory 12-slide solution deck still needs to be authored — see the note
> at the bottom of [`submission/README.md`](submission/README.md).

## Repository structure

```
Drift-Sense/
├── problem statement.pdf      The Applied Materials problem statement this
│                               repository answers.
├── submission/                The graded deliverable (see above) — generator,
│                               localizer, tests, results, references, configs.
└── Team_fab_genr/              A separate, independently-developed DRAM-SEM
                                 generator (GDS layout → rasterization → noise).
                                 Generator-only, no localizer/matcher code;
                                 kept here as supplementary prior work, not
                                 part of the graded submission/ folder above.
```

`Team_fab_genr/` has its own [`README.md`](Team_fab_genr/README.md) and
`requirements.txt`, and runs independently:

```bash
cd Team_fab_genr
pip install -r requirements.txt
python generate_dataset.py --num-samples 20 --split train --output-dir ./output --seed 42
python -m pytest tests/ -v
```
