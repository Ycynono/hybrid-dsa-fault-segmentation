# Expected reference results

Reference machine-readable outputs are distributed in `results/`. Values below are the manuscript targets.

## Thebe test2-test7 macro results

| Method | Exact Dice | 3-pixel tolerant Dice |
|---|---:|---:|
| Locally dip-steered coherence discontinuity | 0.0342 | 0.0572 |
| 3D U-Net | 0.1249 | 0.1934 |
| Hybrid DSA | 0.1468 | 0.2238 |
| SwinUNETR F3-chain | 0.2740 | 0.4284 |

Hybrid DSA exceeds U-Net on all six blocks. The paired exact-Dice mean difference is `0.0219`.

The one-sample dip-steered coherence result has recall `0.9549`, precision `0.0174`, and total six-block runtime `648.5 s` on the reference machine. Its unsteered control is retained separately and achieved exact Dice `0.0343` and tolerant Dice `0.0564`.

## CRACKS expert sections

| Partition | U-Net exact Dice | Hybrid DSA exact Dice | SwinUNETR exact Dice |
|---|---:|---:|---:|
| Audit (20) | 0.0900 | 0.3045 | 0.1919 |
| Sealed reserve (20) | 0.0953 | 0.3112 | 0.1972 |

The reserve Hybrid-minus-Swin exact-Dice difference is `0.1140` with 19 wins in 20 paired sections.

## Smeaheia GN1101 independent sparse-expert ROI

| Method | Precision | Recall | Exact Dice | 3-pixel tolerant Dice | AUPRC | Brier | ECE | Mean/p95 distance (px) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dip-steered coherence | 0.2190 | 0.1782 | 0.1965 | 0.4946 | 0.1986 | 0.1544 | 0.1476 | 12.49 / 43.38 |
| 3D U-Net | 0.2331 | 0.2472 | 0.2399 | 0.5133 | 0.2108 | 0.1765 | 0.1608 | 13.37 / 50.77 |
| Hybrid DSA | 0.2178 | 0.4052 | 0.2833 | 0.5839 | 0.2124 | 0.1630 | 0.1413 | 9.57 / 31.00 |
| SwinUNETR | 0.3011 | 0.1413 | 0.1923 | 0.4853 | 0.2332 | 0.1427 | 0.0885 | 15.79 / 49.01 |

Metrics use only the 297 released expert inline sections and two-dimensional validity corridors. The Hybrid-minus-U-Net macro section exact-Dice difference is `0.0361` with a 95% within-survey section-bootstrap interval of `0.0242-0.0476`. These sections do not constitute independent survey clusters.

Across all 36 released fault objects, macro exact/tolerant Dice is `0.1573/0.4237` for coherence, `0.1436/0.3764` for U-Net, `0.1808/0.4489` for Hybrid DSA and `0.0894/0.2859` for SwinUNETR. Hybrid-minus-U-Net object differences are `0.0372` exact and `0.0725` tolerant Dice, with within-survey fault-object bootstrap intervals of `0.0095-0.0649` and `0.0282-0.1172`.

## Threshold-free and calibration diagnostics

| Dataset | Model | AUPRC | Brier | ECE | Frozen threshold | Best histogram threshold |
|---|---|---:|---:|---:|---:|---:|
| Thebe | U-Net | 0.0718 | 0.0676 | 0.1191 | 0.50 | 0.5825 |
| Thebe | Hybrid DSA | 0.1162 | 0.0233 | 0.0333 | 0.15 | 0.3900 |
| Thebe | SwinUNETR | 0.1966 | 0.0226 | 0.0386 | 0.40 | 0.4550 |
| CRACKS reserve | U-Net | 0.2088 | 0.0589 | 0.0348 | 0.50 | 0.1450 |
| CRACKS reserve | Hybrid DSA | 0.2139 | 0.0599 | 0.0383 | 0.15 | 0.0675 |
| CRACKS reserve | SwinUNETR | 0.2377 | 0.0569 | 0.0205 | 0.40 | 0.1250 |

AUPRC and optimal thresholds are 400-bin histogram approximations. The fixed-threshold manuscript Dice values are exact.

## Efficiency audit

| Model | Parameters | Conv/linear MAC lower bound | Median latency | Throughput | Peak memory |
|---|---:|---:|---:|---:|---:|
| 3D U-Net | 350,809 | 31.633 G | 15.40 ms | 64.92 patches/s | 0.55 GiB |
| Hybrid DSA | 104,787 | 31.078 G | 239.38 ms | 4.18 patches/s | 0.61 GiB |
| SwinUNETR | 4,078,051 | 49.331 G* | 94.69 ms | 10.56 patches/s | 3.27 GiB |

The MAC audit covers convolution, transposed convolution and linear modules. The Swin value excludes attention matrix multiplication and is therefore a lower bound.

## Controlled synthetic sensitivity

The full 11-profile table is in `results/synthetic_ood_sensitivity/summary.json`. Largest paired Dice losses were strong depth-dip for U-Net (`-0.1775`), high white noise for Hybrid DSA (`-0.1729`) and low throw for SwinUNETR (`-0.1540`). These are controlled sensitivity results, not field-generalization estimates.

## Cross-survey stress tests

- FORCE foreground occupancy: U-Net `0`, Hybrid DSA `0.0015%`, SwinUNETR `2.1173%`.
- Delft foreground occupancy: U-Net `25.63%`, Hybrid DSA `45.14%`, SwinUNETR `5.05%`.

These surveys have no independent expert labels for the evaluated ROIs. Do not interpret the values as detection accuracy.

## Numerical tolerance

Evaluation of the distributed checkpoints should reproduce Dice to within `1e-4` when preprocessing, threshold, stride, and data versions match. Runtime is hardware-dependent and is not expected to reproduce exactly.
