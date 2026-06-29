# Hybrid DSA for 3D Seismic Fault Segmentation

Official reproducibility package for **Annotation-domain and cross-survey robustness in 3D seismic fault segmentation: a compact hybrid DSA network**.

The repository contains the complete model source, analytic synthetic-data generator, preprocessing and evaluation code, frozen protocol records, final checkpoints, machine-readable result summaries, and figure scripts. Third-party seismic volumes are not redistributed.

## Evidence design

| Stage | Dataset | Role |
|---|---|---|
| Synthetic pretraining | Analytic generator, seed `20261101` | Controlled dense labels |
| Sparse adaptation | F3 FaultA sticks | Architecture development only |
| Dense field adaptation | Thebe train1-train9 | Matched transfer for all models |
| Selection | Thebe val1-val2 | Checkpoint and threshold selection |
| Frozen test | Thebe test2-test7 | Six independent 3D blocks |
| External expert audit | CRACKS v2 | 20 audit + 20 sealed-reserve sections |
| Calibration stress tests | FORCE and Delft | Label-free external surveys |

Frozen thresholds are U-Net `0.50`, Hybrid DSA `0.15`, and SwinUNETR F3-chain `0.40`. They were selected on Thebe val1-val2 and transferred unchanged.

## Installation

Python 3.11-3.13 is recommended. The reference environment was Windows 11, Python 3.13.9, PyTorch 2.11.0+cu128, CUDA 12.8, and an NVIDIA RTX 5080.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For another CUDA version, install the matching PyTorch wheel first, then run `python -m pip install -r requirements.txt`.

## Five-minute verification

```bash
python scripts/verify_release.py
python synthetic_fault_generator.py --output processed_data/smoke --n-samples 3 --train 1 --val 1 --shape 32,32,32 --qc-count 0
python -m fault_experiments.smoke_test --data-root processed_data/smoke --model unet3d --base-channels 8
python -m fault_experiments.smoke_test --data-root processed_data/smoke --model dsa_hybrid --base-channels 8
```

The verifier checks the three checkpoint SHA-256 hashes against `protocol/FINAL_PROTOCOL_LOCK.json` and verifies that the result files and split manifests exist.

## Full synthetic pretraining

Generate the manuscript dataset:

```bash
python synthetic_fault_generator.py --output processed_data/synthetic_fault_v2_400 --n-samples 400 --train 300 --val 50 --shape 128,128,128 --seed 20261101 --qc-count 6
```

Train the matched 50-epoch models (`batch-size=1`, `base-channels=8`):

```bash
python -m fault_experiments.train_baseline --data-root processed_data/synthetic_fault_v2_400 --output-dir runs/baselines/unet3d_synthetic_e50 --epochs 50 --model unet3d --base-channels 8 --loss focal_tversky
python -m fault_experiments.train_baseline --data-root processed_data/synthetic_fault_v2_400 --output-dir runs/baselines/hybrid_dsa_synthetic_e50 --epochs 50 --model dsa_hybrid --base-channels 8 --loss focal_tversky
python -m fault_experiments.train_baseline --data-root processed_data/synthetic_fault_v2_400 --output-dir runs/baselines/swin_unetr_synthetic_e50 --epochs 50 --model swin_unetr --swin-feature-size 12 --loss focal_tversky --amp
```

The three component ablations use the Hybrid DSA command with one of `--no-depthwise`, `--no-attention`, or `--no-aspp`. The historical fully-depthwise DSA experiments are retained in the source and manuscript result files.

## Field datasets

Download and license details are in [`docs/DATA.md`](docs/DATA.md). After placing source archives in the documented directories:

```bash
python -m fault_experiments.prepare_thebe_dataset --splits train,val,test
python -m fault_experiments.prepare_cracks_external
```

F3, FORCE, and Delft preparation depends on the provider export format. Their dedicated scripts expose the expected paths with `--help` where parameters are supported; exact reference geometry is recorded in `protocol/` and the generated metadata.

## Reproduce frozen evaluations

Evaluate one Thebe block explicitly:

```bash
python -m fault_experiments.evaluate_thebe_block --data-dir processed_data/thebe_official/test/test2 --checkpoint checkpoints/hybrid_dsa_thebe_e8.pt --threshold 0.15 --threshold-source "Thebe val1-val2 only" --output-dir runs/reproduced_thebe/hybrid_dsa/test2 --stride 64
```

Run all frozen Thebe blocks and all three models:

```bash
python -m fault_experiments.run_thebe_final_test
python -m fault_experiments.summarize_thebe_final_test
python -m fault_experiments.plot_thebe_final_statistics
```

Run the preregistered CRACKS audit:

```bash
python -m fault_experiments.evaluate_cracks_audit
python -m fault_experiments.summarize_cracks_audit
python -m fault_experiments.visualize_cracks_audit
```

The sealed reserve script intentionally refuses to overwrite an existing reserve result. Its membership, thresholds, model hashes, and one-time opening policy are stored in `protocol/FINAL_PROTOCOL_LOCK.json`.

For FORCE or Delft, prepare the fixed ROI and run:

```bash
python -m fault_experiments.run_force_frozen_external
python -m fault_experiments.run_delft_frozen_external
```

## Expected results

The reference numerical outputs are in `results/`; see [`docs/EXPECTED_RESULTS.md`](docs/EXPECTED_RESULTS.md). Small floating-point differences may occur across CUDA, cuDNN, and GPU versions. Model ranking and reported thresholds should remain unchanged when the reference checkpoints are evaluated.

## Repository map

```text
checkpoints/       Frozen final checkpoints and hashes
configs/           Training configurations
docs/              Data provenance and full reproduction notes
fault_experiments/ Models, training, preparation, evaluation, and figures
protocol/          Split manifests, protocol lock, environment manifest
results/           Reference JSON/CSV outputs
scripts/           Release verification and convenience commands
tests/             Fast architecture and metric contract tests
```

## Citation and license

Use `CITATION.cff` to cite the software and the associated manuscript. Original code is released under the MIT License. The Thebe, CRACKS, F3, FORCE, and Delft datasets retain their own provider terms; see `docs/DATA.md`.

