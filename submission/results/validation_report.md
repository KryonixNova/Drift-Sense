# Drift-Sense validation report

- checkpoint: `model/production_v3/best.pt`
- device: NVIDIA GeForce RTX 5060 Ti
- python: 3.12.13
- timing method: wall-clock via time.perf_counter() around the single call to model.predict(reference, search); excludes checkpoint/model loading (a one-time cost, reported separately as model_load_time_s) and PNG decode (not part of the localization algorithm itself)

## Noise x geometry conditions

| condition | n | mean err (px) | median err (px) | worst err (px) | pass@5px | pass@4px | pass@2px | pass@1px | pass@0.5px | median runtime (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| noise=normal_geom=normal | 50 | 0.90 | 0.77 | 2.90 | 1.000 | 1.000 | 0.920 | 0.660 | 0.320 | 20.1 |
| noise=harsh_geom=normal | 50 | 42.58 | 1.37 | 856.77 | 0.840 | 0.800 | 0.580 | 0.360 | 0.200 | 20.1 |
| noise=normal_geom=drift | 50 | 0.86 | 0.63 | 2.83 | 1.000 | 1.000 | 0.940 | 0.640 | 0.280 | 20.1 |
| noise=harsh_geom=drift | 50 | 27.08 | 1.35 | 856.75 | 0.840 | 0.820 | 0.580 | 0.360 | 0.240 | 20.2 |

**Pooled (n=200):** median error 0.99px, mean error 17.86px.

`pass@0.5px` is the sub-pixel column: the fraction of pairs localized to better than half a search-image pixel.

**Read the median, not the mean.** The error distribution is strongly bimodal: most pairs land sub-pixel, while 5/200 (2.5%) are gross mis-localizations past 50px — the decoder locking onto a lattice twin rather than the true cell, which puts it hundreds of pixels away, not a few. Excluding those, the mean over the remaining 195 pairs is 1.76px. A single mean over the mixture describes neither group; the pass-rate columns and the median are the honest summaries.

## Stratified results

The spec asks for results across target positions, scales and rotations as well as noise levels. Same runs as above, re-sliced along each of those axes.

Because of the tail described above, compare the **median** and **pass-rate** columns across buckets — a single gross failure moves a bucket's mean by tens of pixels and can make an axis that carries no real signal (scale, rotation) look like it does.

### By target position

*all conditions; distance from target centre to nearest search-image border.*

| bucket | n | mean err (px) | median err (px) | worst err (px) | pass@5px | pass@4px | pass@2px | pass@1px | pass@0.5px |
|---|---|---|---|---|---|---|---|---|---|
| edge (<150px) | 76 | 45.05 | 1.41 | 856.77 | 0.816 | 0.789 | 0.592 | 0.355 | 0.211 |
| mid (150-300px) | 76 | 1.57 | 1.13 | 9.81 | 0.974 | 0.961 | 0.763 | 0.408 | 0.211 |
| centre (>=300px) | 48 | 0.60 | 0.53 | 1.47 | 1.000 | 1.000 | 1.000 | 0.896 | 0.417 |

### By barrel distortion

*all conditions; magnitude of the radial lens warp applied to the search image (non-zero under imaging_noise_profile=harsh only). Displacement grows with r^2, so this is the same effect the target-position slice measures indirectly.*

| bucket | n | mean err (px) | median err (px) | worst err (px) | pass@5px | pass@4px | pass@2px | pass@1px | pass@0.5px |
|---|---|---|---|---|---|---|---|---|---|
| none (k < 0.01) | 116 | 0.83 | 0.62 | 2.90 | 1.000 | 1.000 | 0.940 | 0.690 | 0.345 |
| mild (0.01-0.03) | 18 | 40.55 | 2.18 | 347.17 | 0.778 | 0.778 | 0.444 | 0.167 | 0.111 |
| moderate (0.03-0.06) | 32 | 29.79 | 1.30 | 820.86 | 0.750 | 0.688 | 0.562 | 0.375 | 0.188 |
| severe (>= 0.06) | 34 | 52.69 | 2.31 | 856.77 | 0.882 | 0.853 | 0.471 | 0.176 | 0.118 |

### By reference scale ratio

*geometric_profile=drift only (nominal 10:1 elsewhere); reference magnification ratio, 9:1-11:1 sweep.*

| bucket | n | mean err (px) | median err (px) | worst err (px) | pass@5px | pass@4px | pass@2px | pass@1px | pass@0.5px |
|---|---|---|---|---|---|---|---|---|---|
| 0.90-0.95 | 28 | 1.31 | 0.90 | 4.86 | 1.000 | 0.964 | 0.714 | 0.536 | 0.357 |
| 0.95-1.00 | 17 | 0.94 | 1.04 | 1.84 | 1.000 | 1.000 | 1.000 | 0.471 | 0.294 |
| 1.00-1.05 | 31 | 30.89 | 1.13 | 856.75 | 0.839 | 0.839 | 0.645 | 0.484 | 0.226 |
| 1.05-1.10 | 24 | 16.11 | 1.01 | 347.17 | 0.875 | 0.875 | 0.792 | 0.500 | 0.167 |

### By reference rotation

*geometric_profile=drift only (0deg elsewhere); absolute reference rotation.*

| bucket | n | mean err (px) | median err (px) | worst err (px) | pass@5px | pass@4px | pass@2px | pass@1px | pass@0.5px |
|---|---|---|---|---|---|---|---|---|---|
| 0.0-0.5 deg | 23 | 1.41 | 0.78 | 7.42 | 0.957 | 0.957 | 0.783 | 0.522 | 0.174 |
| 0.5-1.0 deg | 20 | 1.82 | 0.87 | 8.43 | 0.900 | 0.900 | 0.750 | 0.550 | 0.300 |
| 1.0-1.5 deg | 32 | 40.25 | 1.14 | 856.75 | 0.875 | 0.844 | 0.781 | 0.469 | 0.219 |
| 1.5-2.0 deg | 25 | 1.60 | 1.04 | 9.75 | 0.960 | 0.960 | 0.720 | 0.480 | 0.360 |

## Failure case

![failure case](failure_case.png)

Worst case (856.8px error) occurred under imaging_noise_profile=harsh, geometric_profile=normal, with model confidence=0.122. The target sat 70px from the nearest search-image border, under a barrel-distortion coefficient of k=-0.074. Those are the two settings that carry the error in this run: mean error runs 0.60px for targets in the 'centre (>=300px)' band against 45.05px for 'edge (<150px)', and 0.83px at 'none (k < 0.01)' barrel distortion against 52.69px at 'severe (>= 0.06)'. Those two axes are not independent: barrel distortion displaces a feature radially by an amount growing with r^2, so it is near-zero at the image centre and largest exactly where the edge-position bucket's targets sit. Ablating the barrel term alone (holding the canvases, crops, labels and every other noise source fixed) removes most of the harsh-profile error, which makes it the dominant driver here rather than repeated-pattern ambiguity. Note that ground truth is computed from pre-distortion canvas geometry and is never corrected for this warp, so under harsh conditions these numbers conflate localization error with label displacement; see the limitation note in README section 12.
