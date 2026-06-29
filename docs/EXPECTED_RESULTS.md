# Expected reference results

Reference machine-readable outputs are distributed in `results/`. Values below are the manuscript targets.

## Thebe test2-test7 macro results

| Method | Exact Dice | 3-pixel tolerant Dice |
|---|---:|---:|
| Local coherence discontinuity | 0.0343 | 0.0564 |
| 3D U-Net | 0.1249 | 0.1934 |
| Hybrid DSA | 0.1468 | 0.2238 |
| SwinUNETR F3-chain | 0.2740 | 0.4284 |

Hybrid DSA exceeds U-Net on all six blocks. The paired exact-Dice mean difference is `0.0219`.

## CRACKS expert sections

| Partition | U-Net exact Dice | Hybrid DSA exact Dice | SwinUNETR exact Dice |
|---|---:|---:|---:|
| Audit (20) | 0.0900 | 0.3045 | 0.1919 |
| Sealed reserve (20) | 0.0953 | 0.3112 | 0.1972 |

The reserve Hybrid-minus-Swin exact-Dice difference is `0.1140` with 19 wins in 20 paired sections.

## Cross-survey stress tests

- FORCE foreground occupancy: U-Net `0`, Hybrid DSA `0.0015%`, SwinUNETR `2.1173%`.
- Delft foreground occupancy: U-Net `25.63%`, Hybrid DSA `45.14%`, SwinUNETR `5.05%`.

These surveys have no independent expert labels for the evaluated ROIs. Do not interpret the values as detection accuracy.

## Numerical tolerance

Evaluation of the distributed checkpoints should reproduce Dice to within `1e-4` when preprocessing, threshold, stride, and data versions match. Runtime is hardware-dependent and is not expected to reproduce exactly.

