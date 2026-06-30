# Frozen Smeaheia external 3D validation protocol

## Decision

Smeaheia is the primary candidate for the missing independent three-dimensional expert validation. The official CO2DataShare release (DOI `10.11582/2021.00012`) contains the GN1101 and TNE01 3D SEG-Y surveys and Petrel ASCII fault sticks. GN1101, rather than TNE01, is the interpretation survey for the released fault framework.

The resource can address the missing independent-survey evidence only after coordinate registration and label-coverage checks pass. The fault sticks are sparse expert interpretations, not a dense voxelwise inventory. Uninterpreted voxels must not be treated as fault-negative ground truth.

## Pre-result lock

1. Use the released GN1101 seismic cube and released fault sticks without manual alteration.
2. Preserve the U-Net, Hybrid DSA and SwinUNETR checkpoints and Thebe-selected thresholds unchanged.
3. Choose the evaluation domain from expert-stick coverage and SEG-Y geometry before opening any model prediction.
4. Exclude TNE01 from accuracy evaluation because the provider states that it was not used for the interpretation study.
5. Rasterize each expert stick into a one-voxel centreline and define a symmetric validity corridor in physical units.
6. Report exact and tolerance-aware precision, recall, Dice, mean surface distance and 95th-percentile surface distance inside the validity domain.
7. Report results both per fault object and as a pooled sparse-label audit. Use object/bootstrap intervals only; do not call them independent-survey confidence intervals.
8. Compare the same ROI with dip-steered coherence, U-Net, Hybrid DSA and SwinUNETR.
9. Render expert sticks and unmodified probability volumes in the same 3D coordinates. Mesh filtering and decimation are display-only and must be audited.
10. If coordinate reference system, vertical domain or trace-header mapping cannot be verified, classify the experiment as unusable rather than forcing registration.

## Data-quality gates

| Gate | Pass condition | Failure consequence |
|---|---|---|
| Provenance | Files originate from the official DOI release and hashes are recorded. | Do not cite or evaluate. |
| Survey identity | Fault-stick coordinates overlap GN1101, not merely the broader Smeaheia area. | Do not claim independent 3D validation. |
| Horizontal registration | Petrel X/Y picks map to SEG-Y trace coordinates with residual below half a trace spacing. | Investigate CRS/header fields; no metrics. |
| Vertical registration | Stick Z unit/domain matches SEG-Y time or is converted using released velocity information. | No voxelwise or surface-distance metrics. |
| Coverage | At least three separated fault objects and a nontrivial interpreted depth/time range overlap the cube. | Treat as a case study only. |
| Label semantics | Sparse validity corridors are used; outside-corridor voxels are ignored. | Metrics are invalid due to false-negative contamination. |
| Frozen inference | Model hashes and thresholds match `FINAL_PROTOCOL_LOCK.json`. | Rerun from frozen checkpoints. |

## Claim boundary

A successful result adds one independent field-survey cluster with expert-derived 3D geometry. It does not create a dense exhaustive fault truth, does not establish population-level generalization, and does not replace a future multi-interpreter study. It is nevertheless materially stronger than label-free FORCE/Delft visualization and directly addresses the editor's request for testing across multiple real datasets or conditions.
