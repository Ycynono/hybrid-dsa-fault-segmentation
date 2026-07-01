# Data provenance and local layout

The repository does not redistribute third-party seismic volumes. Download each resource from its original provider, accept its terms, and retain its citation.

## Thebe

- Source: Harvard Dataverse, DOI `10.7910/DVN/YBYGBK`.
- Papers: An et al. (2021a, 2021b), listed in the manuscript.
- Role: train1-train9 adaptation; val1-val2 checkpoint/threshold selection; test2-test7 frozen evaluation.
- Local source directory: `external_data/Thebe/complete/`.
- Prepared directory: `processed_data/thebe_official/`.

```bash
python download_thebe_complete.py
python -m fault_experiments.prepare_thebe_dataset --download-root external_data/Thebe/complete --output-root processed_data/thebe_official --splits train,val,test
```

Verify the generated manifest against `protocol/thebe_manifest.json` before evaluation.

## CRACKS v2

- Source: Zenodo, DOI `10.5281/zenodo.13926822`, CC BY 4.0.
- Role: external expert interpretations; no label used for training or threshold selection.
- Local source directory: `external_data/CRACKS/` containing the original image and segmentation archives.
- Prepared directory: `processed_data/cracks_external_v1/`.

```bash
python -m fault_experiments.prepare_cracks_external
```

The fixed membership is in `protocol/cracks_expert_split.json`. The reserve was opened once only after `protocol/FINAL_PROTOCOL_LOCK.json` was written.

## F3 Demo 2023

- Source: dGB Earth Sciences, TerraNubis: <https://terranubis.com/datainfo/F3-Demo-2023>.
- Role: sparse FaultA development adaptation and CRACKS image domain.
- Provider metadata records Creative Commons 3.0.
- Do not treat non-annotated voxels outside validity corridors as negatives.

The sparse interpretation is converted by `fault_experiments.prepare_f3_faulta_benchmark`; the manifest stores coordinate transforms and stick membership.

## FORCE ML Competition 2020

- Source: FORCE, Geoscience Australia, and dGB Earth Sciences, TerraNubis: <https://terranubis.com/datainfo/FORCE-ML-Competition-2020>.
- Survey: Ichthys, offshore northwest Australia; provider metadata records CC BY 4.0.
- Role: label-free frozen calibration stress test.
- Reference ROI: `384 x 512 x 128`, fixed before comparison.

No expert label exists for the evaluated ROI. Occupancy and connected-component results are diagnostics, not accuracy.

## Delft

- Source: dGB Earth Sciences, TerraNubis: <https://terranubis.com/datainfo/Delft>.
- Provider metadata records Creative Commons 3.0.
- Role: label-free frozen calibration stress test and comparison with the provided thinned fault-likelihood attribute.
- Prepared directory: `processed_data/delft_external_center/`.
- Reference ROI: inlines 2533-2916, crosslines 3181-3564, 1.644-2.152 s.

```bash
python -m fault_experiments.prepare_delft_external --help
python -m fault_experiments.run_delft_frozen_external
```

The TFL volume is an algorithmic attribute, not an independent interpretation. Its overlap with network masks must be reported as attribute agreement only.

## Smeaheia GN1101 independent 3D expert validation

- Official source: CO2DataShare, DOI `10.11582/2021.00012`.
- Required components: `Fault sticks`, `Reports`, and `Seismic 3D surveys`.
- Interpretation survey: GN1101. The provider states that TNE01 was not used for the feasibility-study interpretation.
- Fault format: Petrel ASCII; seismic format: SEG-Y.
- License: Smeaheia Dataset License, based on CC BY 4.0 with a no-sale condition. Attribute Gassnova and Equinor and link the license when sharing derived material.
- Local source directory: `external_data/smeaheia/`.

The official download form records country/territory and institution. The script requires these as command-line arguments and never places them in source control:

```bash
python scripts/download_smeaheia.py fault_sticks reports seismic_3d --country "<country>" --affiliation "<institution>"
```

The downloaded GN1101 file and fault-stick interpretation should have SHA-256 hashes `1229da79662eed4edb62beca8582088d59a431d6f768271abcf14cbda76ba800` and `c889e8e067e2585f98ce32d173c913e464dace9b702c3335a710a81cd7b5a78f`. Run:

```bash
python -m fault_experiments.audit_smeaheia_dataset --require-segy
python -m fault_experiments.prepare_smeaheia_benchmark
```

The reference audit matched 100% of grid-indexed picks, with horizontal p95 residual `6.06 m` (below half the `12.48 m` trace spacing) and vertical p95 residual `1.90 ms` (below half the `4 ms` sampling). The deterministic `384 x 512 x 640` ROI contains 4,691 points from 36 faults on 297 expert-active inline sections. This is independent sparse 3D expert geometry, not dense voxel truth. Petrel X/Y/Z picks must pass all gates in `protocol/SMEAHEIA_VALIDATION_PROTOCOL.md`, and voxels outside expert-stick corridors must be ignored.

## Integrity rules

1. Never tune on Thebe test2-test7 or either CRACKS partition.
2. Keep the three frozen thresholds unchanged for all external evaluations.
3. Do not redistribute provider data in forks or releases.
4. Retain generated manifests, checkpoint hashes, and command logs with every reproduced run.
5. Do not report Smeaheia metrics unless its horizontal and vertical registration gates pass.
