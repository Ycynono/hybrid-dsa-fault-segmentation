# Reproduce revised figures and diagnostics

Run commands from the repository root after preparing the licensed datasets according to `DATA.md`. The commands below never download or redistribute third-party seismic volumes.

## Frozen probability and threshold diagnostics

```bash
python -m fault_experiments.analyze_probability_calibration
```

Expected outputs:

- `runs/probability_calibration/summary.json`
- `runs/probability_calibration/threshold_curves.csv`
- `runs/probability_calibration/threshold_and_pr_curves.png`

The command reads the frozen full-volume probability sums and never changes checkpoints or primary thresholds.

## Controlled synthetic sensitivity

```bash
python -m fault_experiments.evaluate_synthetic_ood_sensitivity
```

Expected outputs:

- `runs/synthetic_ood_sensitivity/summary.json`
- `runs/synthetic_ood_sensitivity/per_sample_metrics.csv`
- `runs/synthetic_ood_sensitivity/synthetic_ood_sensitivity.png`

All 11 profiles reuse the same eight seeds. The profile changes include frequency, 90-degree phase, white and correlated noise, throw, curvature, depth-dip, fault-zone attenuation and lateral sampling.

## Cross-survey mechanism diagnostics

```bash
python -m fault_experiments.analyze_cross_survey_mechanisms
```

Expected outputs:

- `runs/cross_survey_mechanisms/survey_domain_descriptors.csv`
- `runs/cross_survey_mechanisms/mechanism_diagnostics.json`
- `runs/cross_survey_mechanisms/cross_survey_mechanism_attribution.png`

The script uses deterministic sampled texture descriptors across Synthetic, F3, Thebe, FORCE and Delft, followed by positive 3D gradient-weighted attribution on fixed central 64-cube patches for Hybrid DSA and SwinUNETR. Frequency is normalized to Nyquist because all physical sample intervals are not verified. The lineation statistic is a texture proxy, and attribution identifies model sensitivity rather than fault accuracy or geological causality.

## Locally dip-steered coherence

```bash
python -m fault_experiments.evaluate_coherence_baseline calibrate --output-dir runs/dip_steered_coherence_baseline --max-dip-shift 1
python -m fault_experiments.evaluate_coherence_baseline evaluate --output-dir runs/dip_steered_coherence_baseline --max-dip-shift 1
python -m fault_experiments.visualize_thebe_orthogonal_3d --coherence-run runs/dip_steered_coherence_baseline
```

Only the threshold is selected on Thebe val1-val2. Test2-test7 use that threshold unchanged. Use `--max-dip-shift 0` to reproduce the unsteered control.

## Three-dimensional display audit

```bash
python -m fault_experiments.visualize_true_3d_fault_surfaces
```

The JSON companions record raw voxels, thinning retention, display-grid reduction, components before and after filtering, mesh faces and VTP paths. These steps are display-only; full-resolution masks remain the metric source.

## Smeaheia independent expert comparison

```bash
python -m fault_experiments.audit_smeaheia_dataset --require-segy
python -m fault_experiments.prepare_smeaheia_benchmark
python -m fault_experiments.run_smeaheia_frozen_external
python -m fault_experiments.run_smeaheia_coherence
python -m fault_experiments.evaluate_smeaheia_benchmark
python -m fault_experiments.evaluate_smeaheia_fault_objects
python -m fault_experiments.visualize_smeaheia_comparison
```

Expected figures are `smeaheia_metric_comparison.png`, `smeaheia_expert_method_sections.png`, and `smeaheia_true_3d_fault_surfaces.png`. The three-dimensional network and coherence meshes are clipped to the same expert validity corridor for display comparability. All reported metrics use unmodified arrays within that corridor.

## Efficiency and evidence hierarchy

```bash
python -m fault_experiments.benchmark_model_inference --output runs/efficiency/inference_benchmark.json
python -m fault_experiments.audit_evidence_hierarchy
```

Latency, throughput and memory are hardware-dependent. MAC values are module-hook lower bounds and exclude attention matrix multiplication. The evidence audit explicitly distinguishes within-survey units from independent survey clusters.

## Other manuscript figures

```bash
python -m fault_experiments.plot_paper_schematics
python -m fault_experiments.plot_thebe_final_statistics
python -m fault_experiments.visualize_thebe_final_sections
python -m fault_experiments.visualize_cracks_audit
python -m fault_experiments.visualize_cracks_reserve_cases
python -m fault_experiments.visualize_force_frozen_external
python -m fault_experiments.visualize_delft_frozen_external
python -m fault_experiments.plot_cross_dataset_evidence
```

Reference outputs and their machine-readable summaries are under `results/`. Checkpoint hashes and frozen thresholds are under `protocol/`.
