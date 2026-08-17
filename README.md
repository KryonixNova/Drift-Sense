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
