import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec

from fault_experiments.infer_real_volume import load_model


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "cross_survey_mechanisms"
RNG_SEED = 20260702

SURVEYS = {
    "Synthetic": sorted((ROOT / "processed_data/synthetic_fault_v2_400/train").glob("syn_*/amplitude.npy"))[:8],
    "F3": [ROOT / "processed_data/real_subvolumes/f3_main_384x512x128/amplitude_norm.npy"],
    "Thebe": [
        ROOT / f"processed_data/thebe_official/train/train{i}/amplitude_norm.npy"
        for i in range(1, 4)
    ],
    "FORCE": [ROOT / "processed_data/real_subvolumes/force_field_mid_384x512x128/amplitude_norm.npy"],
    "Delft": [ROOT / "processed_data/delft_external_center/amplitude_norm.npy"],
}

ATTRIBUTION_SURVEYS = {
    "FORCE": SURVEYS["FORCE"][0],
    "Delft": SURVEYS["Delft"][0],
}

MODELS = {
    "Hybrid DSA": ROOT / "runs/thebe_adaptation/dsa_hybrid_replay_e8/best.pt",
    "SwinUNETR": ROOT / "runs/thebe_adaptation/swin_unetr_f3chain_e8/best.pt",
}

METRIC_LABELS = {
    "spectral_centroid_nyquist": "Spectral\ncentroid",
    "high_frequency_power_fraction": "High-frequency\npower",
    "lateral_incoherence": "Lateral\nincoherence",
    "deep_to_shallow_highpass_ratio": "Deep/shallow\nhigh-pass",
    "lateral_to_vertical_gradient_ratio": "Lateral/vertical\ngradient",
    "axis_lineation_proxy": "Axis-lineation\nproxy",
}


def normalize_amplitude(array):
    array = np.asarray(array, dtype=np.float32)
    if float(np.nanmin(array)) >= -0.01:
        array = 2.0 * array - 1.0
    return np.clip(array, -1.0, 1.0)


def sampled_traces(paths, rng, count=1536):
    traces = []
    per_path = max(1, count // len(paths))
    for path in paths:
        volume = np.load(path, mmap_mode="r")
        i = rng.integers(0, volume.shape[0], size=per_path)
        j = rng.integers(0, volume.shape[1], size=per_path)
        traces.append(normalize_amplitude(volume[i, j, :]))
    return np.concatenate(traces, axis=0)


def trace_spectrum_metrics(traces):
    traces = traces - np.mean(traces, axis=1, keepdims=True)
    window = np.hanning(traces.shape[1]).astype(np.float32)
    spectra = np.fft.rfft(traces * window[None, :], axis=1)
    power = np.abs(spectra) ** 2
    power[:, 0] = 0.0
    power /= np.maximum(power.sum(axis=1, keepdims=True), 1e-12)
    frequency = np.linspace(0.0, 1.0, power.shape[1], dtype=np.float64)
    centroid = np.sum(power * frequency[None, :], axis=1)
    high_fraction = power[:, frequency >= 0.5].sum(axis=1)

    difference = np.diff(traces, axis=1)
    third = max(1, difference.shape[1] // 3)
    shallow = np.mean(difference[:, :third] ** 2, axis=1)
    deep = np.mean(difference[:, -third:] ** 2, axis=1)
    deep_ratio = deep / np.maximum(shallow, 1e-12)
    return {
        "spectral_centroid_nyquist": float(np.median(centroid)),
        "high_frequency_power_fraction": float(np.median(high_fraction)),
        "deep_to_shallow_highpass_ratio": float(np.median(deep_ratio)),
    }


def neighbor_metrics(paths, rng, count=1024):
    correlations = []
    gradient_ratios = []
    per_path = max(1, count // (2 * len(paths)))
    for path in paths:
        volume = np.load(path, mmap_mode="r")
        for axis in (0, 1):
            upper = volume.shape[axis] - 1
            i = rng.integers(0, volume.shape[0], size=per_path)
            j = rng.integers(0, volume.shape[1], size=per_path)
            if axis == 0:
                i = np.minimum(i, upper - 1)
                a = normalize_amplitude(volume[i, j, :])
                b = normalize_amplitude(volume[i + 1, j, :])
            else:
                j = np.minimum(j, upper - 1)
                a = normalize_amplitude(volume[i, j, :])
                b = normalize_amplitude(volume[i, j + 1, :])
            a = a - a.mean(axis=1, keepdims=True)
            b = b - b.mean(axis=1, keepdims=True)
            denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
            correlations.extend((np.sum(a * b, axis=1) / np.maximum(denominator, 1e-12)).tolist())

        for _ in range(4):
            d0 = min(64, volume.shape[0])
            d1 = min(64, volume.shape[1])
            d2 = min(128, volume.shape[2])
            starts = [
                int(rng.integers(0, max(1, n - d + 1)))
                for n, d in zip(volume.shape, (d0, d1, d2))
            ]
            cube = normalize_amplitude(
                volume[
                    starts[0] : starts[0] + d0,
                    starts[1] : starts[1] + d1,
                    starts[2] : starts[2] + d2,
                ]
            )
            lateral = 0.5 * (
                np.median(np.abs(np.diff(cube, axis=0)))
                + np.median(np.abs(np.diff(cube, axis=1)))
            )
            vertical = np.median(np.abs(np.diff(cube, axis=2)))
            gradient_ratios.append(float(lateral / max(vertical, 1e-12)))
    return {
        "lateral_incoherence": float(1.0 - np.median(correlations)),
        "lateral_to_vertical_gradient_ratio": float(np.median(gradient_ratios)),
    }


def axis_lineation_metric(paths, rng, slices_per_path=6):
    values = []
    for path in paths:
        volume = np.load(path, mmap_mode="r")
        accepted = 0
        attempts = 0
        while accepted < slices_per_path and attempts < slices_per_path * 20:
            attempts += 1
            t = int(rng.integers(0, volume.shape[2]))
            d0 = min(256, volume.shape[0])
            d1 = min(256, volume.shape[1])
            i = int(rng.integers(0, max(1, volume.shape[0] - d0 + 1)))
            j = int(rng.integers(0, max(1, volume.shape[1] - d1 + 1)))
            image = normalize_amplitude(volume[i : i + d0, j : j + d1, t])
            image = image - np.mean(image)
            if float(np.std(image)) < 1e-4:
                continue
            window = np.outer(np.hanning(d0), np.hanning(d1)).astype(np.float32)
            power = np.abs(np.fft.fftshift(np.fft.fft2(image * window))) ** 2
            ky, kx = np.meshgrid(
                np.fft.fftshift(np.fft.fftfreq(d1)),
                np.fft.fftshift(np.fft.fftfreq(d0)),
            )
            radius = np.sqrt(kx * kx + ky * ky)
            valid = (radius >= 0.06) & (radius <= 0.40)
            axis_band = valid & ((np.abs(kx) <= 0.025) | (np.abs(ky) <= 0.025))
            observed = power[axis_band].sum() / max(power[valid].sum(), 1e-12)
            expected = axis_band.sum() / max(valid.sum(), 1)
            values.append(float(observed / max(expected, 1e-12)))
            accepted += 1
        if accepted < slices_per_path:
            raise RuntimeError(f"Could not sample enough non-constant time slices from {path}")
    return float(np.median(values))


def survey_descriptors():
    rows = []
    for index, (survey, paths) in enumerate(SURVEYS.items()):
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {survey} inputs: {missing}")
        rng = np.random.default_rng(RNG_SEED + index)
        metrics = {"survey": survey}
        metrics.update(trace_spectrum_metrics(sampled_traces(paths, rng)))
        metrics.update(neighbor_metrics(paths, rng))
        metrics["axis_lineation_proxy"] = axis_lineation_metric(paths, rng)
        rows.append(metrics)
        print(f"descriptors complete: {survey}", flush=True)
    return rows


def center_patch(path, size=64):
    volume = np.load(path, mmap_mode="r")
    starts = [(n - size) // 2 for n in volume.shape]
    patch = volume[
        starts[0] : starts[0] + size,
        starts[1] : starts[1] + size,
        starts[2] : starts[2] + size,
    ]
    return normalize_amplitude(patch), starts


def grad_cam(model, tensor, layers):
    captured = {}
    handles = []

    def hook(name):
        def save(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise TypeError(f"Attribution layer {name} did not return a tensor")
            output.retain_grad()
            captured[name] = output

        return save

    modules = dict(model.named_modules())
    for label, module_name in layers.items():
        handles.append(modules[module_name].register_forward_hook(hook(label)))

    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    count = max(1, logits.numel() // 1000)
    score = torch.topk(logits.reshape(-1), count).values.mean()
    score.backward()

    cams = {}
    for label, activation in captured.items():
        gradient = activation.grad
        weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tensor.shape[2:], mode="trilinear", align_corners=False)
        cam = cam[0, 0].detach().float().cpu().numpy()
        high = float(np.percentile(cam, 99.5))
        if high <= 0:
            high = float(np.max(cam))
        cams[label] = np.clip(cam / max(high, 1e-12), 0.0, 1.0)

    probability = torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy()
    for handle in handles:
        handle.remove()
    return cams, probability, float(score.detach().cpu())


def attributions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_names = {
        "Hybrid DSA": {"shallow": "enc1", "deep": "bottleneck"},
        "SwinUNETR": {"shallow": "encoder1", "deep": "encoder10"},
    }
    results = {}
    for model_name, checkpoint in MODELS.items():
        model, _ = load_model(checkpoint, device)
        results[model_name] = {}
        for survey, path in ATTRIBUTION_SURVEYS.items():
            patch, starts = center_patch(path)
            tensor = torch.from_numpy(np.ascontiguousarray(patch[None, None])).to(device)
            cams, probability, score = grad_cam(model, tensor, layer_names[model_name])
            results[model_name][survey] = {
                "amplitude": patch,
                "probability": probability,
                "cams": cams,
                "starts": starts,
                "top_logit_score": score,
                "probability_mean": float(probability.mean()),
                "probability_p99": float(np.percentile(probability, 99)),
            }
            print(f"attribution complete: {model_name} / {survey}", flush=True)
            del tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def robust_zscore(rows, metric_names):
    matrix = np.asarray([[row[name] for name in metric_names] for row in rows], dtype=np.float64)
    median = np.median(matrix, axis=0, keepdims=True)
    mad = np.median(np.abs(matrix - median), axis=0, keepdims=True)
    return np.clip((matrix - median) / np.maximum(1.4826 * mad, 1e-9), -3.0, 3.0)


def overlay_cam(amplitude, cam):
    base = plt.get_cmap("gray")(np.clip((amplitude + 1.0) / 2.0, 0.0, 1.0))
    color = plt.get_cmap("magma")(cam)
    alpha = (0.72 * np.clip((cam - 0.15) / 0.85, 0.0, 1.0))[..., None]
    base[..., :3] = (1.0 - alpha) * base[..., :3] + alpha * color[..., :3]
    return base


def make_figure(rows, attribution_results):
    metric_names = list(METRIC_LABELS)
    zscore = robust_zscore(rows, metric_names)
    fig = plt.figure(figsize=(15.5, 10.2), dpi=220, facecolor="white")
    grid = GridSpec(3, 5, figure=fig, height_ratios=(1.35, 2.2, 2.2), hspace=0.34, wspace=0.08)

    heat_ax = fig.add_subplot(grid[0, :])
    image = heat_ax.imshow(zscore, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    heat_ax.set_yticks(range(len(rows)), [row["survey"] for row in rows], fontsize=9)
    heat_ax.set_xticks(range(len(metric_names)), [METRIC_LABELS[name] for name in metric_names], fontsize=8)
    heat_ax.set_title("a  Data-domain descriptors (robust z-score across five domains)", loc="left", fontsize=11, fontweight="bold")
    for i in range(zscore.shape[0]):
        for j in range(zscore.shape[1]):
            heat_ax.text(j, i, f"{zscore[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(zscore[i, j]) > 1.7 else "black")
    colorbar = fig.colorbar(image, ax=heat_ax, fraction=0.018, pad=0.015)
    colorbar.set_label("Robust z-score", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    columns = [
        (None, None, "Seismic amplitude"),
        ("Hybrid DSA", "shallow", "Hybrid shallow"),
        ("Hybrid DSA", "deep", "Hybrid deep"),
        ("SwinUNETR", "shallow", "Swin shallow"),
        ("SwinUNETR", "deep", "Swin deep"),
    ]
    for row_index, survey in enumerate(("FORCE", "Delft"), start=1):
        hybrid = attribution_results["Hybrid DSA"][survey]
        middle = hybrid["amplitude"].shape[0] // 2
        seismic = hybrid["amplitude"][middle].T
        for column_index, (model_name, depth, title) in enumerate(columns):
            ax = fig.add_subplot(grid[row_index, column_index])
            if model_name is None:
                ax.imshow((seismic + 1.0) / 2.0, cmap="gray", vmin=0, vmax=1, aspect="auto")
            else:
                cam = attribution_results[model_name][survey]["cams"][depth][middle].T
                ax.imshow(overlay_cam(seismic, cam), aspect="auto")
            if row_index == 1:
                ax.set_title(title, fontsize=9, pad=5)
            if column_index == 0:
                prefix = "b" if survey == "FORCE" else "c"
                ax.set_ylabel(f"{prefix}  {survey}\nTime sample", fontsize=9, fontweight="bold")
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("Lateral sample", fontsize=8)
            ax.tick_params(labelsize=7, length=2)

    fig.suptitle(
        "Cross-survey activation diagnostics: sample-domain descriptors and 3D gradient-weighted feature attribution",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    output = OUTPUT / "cross_survey_mechanism_attribution.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def serializable_attributions(results):
    payload = {
        "method": (
            "Grad-CAM-style positive attribution of the mean top 0.1% output logits on a fixed central 64^3 patch. "
            "Maps explain model sensitivity and are not accuracy or causal-geology evidence."
        ),
        "models": {},
    }
    for model_name, surveys in results.items():
        payload["models"][model_name] = {}
        for survey, values in surveys.items():
            payload["models"][model_name][survey] = {
                "patch_starts": values["starts"],
                "patch_shape": list(values["amplitude"].shape),
                "top_logit_score": values["top_logit_score"],
                "probability_mean": values["probability_mean"],
                "probability_p99": values["probability_p99"],
            }
    return payload


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = survey_descriptors()
    metric_names = list(METRIC_LABELS)
    with (OUTPUT / "survey_domain_descriptors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["survey", *metric_names])
        writer.writeheader()
        writer.writerows(rows)

    attribution_results = attributions()
    figure = make_figure(rows, attribution_results)
    payload = {
        "scope": "Mechanism-oriented diagnostics for frozen cross-survey activation failure",
        "descriptor_caveat": (
            "Frequencies are normalized to Nyquist because physical sample intervals are not verified for every survey. "
            "Axis lineation is a directional image-texture proxy and cannot by itself establish acquisition footprint."
        ),
        "descriptors": rows,
        "attribution": serializable_attributions(attribution_results),
        "figure": str(figure.relative_to(ROOT)),
        "seed": RNG_SEED,
    }
    (OUTPUT / "mechanism_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
