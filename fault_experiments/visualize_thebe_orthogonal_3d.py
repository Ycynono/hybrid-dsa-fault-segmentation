from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from fault_experiments.visualize_thebe_3d import reconstruction_weights


ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("U-Net", "unet3d", 0.50),
    ("Hybrid DSA", "dsa_hybrid_replay", 0.15),
    ("SwinUNETR", "swin_unetr_f3chain", 0.40),
]


def structured_plane(axis, fixed, a_values, b_values, image, exaggeration):
    aa, bb = np.meshgrid(a_values, b_values, indexing="ij")
    if axis == "crossline":
        xx = np.full_like(aa, fixed * exaggeration, dtype=np.float32)
        yy = aa.astype(np.float32)
        zz = -bb.astype(np.float32)
    elif axis == "inline":
        xx = aa.astype(np.float32) * exaggeration
        yy = np.full_like(aa, fixed, dtype=np.float32)
        zz = -bb.astype(np.float32)
    else:
        xx = aa.astype(np.float32) * exaggeration
        yy = bb.astype(np.float32)
        zz = np.full_like(aa, -fixed, dtype=np.float32)
    grid = pv.StructuredGrid(xx, yy, zz)
    grid["amplitude"] = np.asarray(image, dtype=np.float32).ravel(order="F")
    return grid


def fault_points(axis, fixed, a_values, b_values, values, threshold, exaggeration):
    mask = values >= threshold
    ia, ib = np.nonzero(mask)
    if axis == "crossline":
        coords = np.column_stack(
            [np.full(len(ia), fixed * exaggeration), a_values[ia], -b_values[ib]]
        )
    elif axis == "inline":
        coords = np.column_stack(
            [a_values[ia] * exaggeration, np.full(len(ia), fixed), -b_values[ib]]
        )
    else:
        coords = np.column_stack(
            [a_values[ia] * exaggeration, b_values[ib], np.full(len(ia), -fixed)]
        )
    cloud = pv.PolyData(coords.astype(np.float32))
    cloud["fault_probability"] = np.asarray(values[mask], dtype=np.float32)
    return cloud


def extract_amplitude_planes(amplitude, crossline, inline, sample, inline_slice, sample_slice):
    return {
        "crossline": np.asarray(amplitude[crossline, inline_slice, sample_slice]),
        "inline": np.asarray(amplitude[:, inline, sample_slice]),
        "sample": np.asarray(amplitude[:, inline_slice, sample]),
    }


def extract_label_planes(label, crossline, inline, sample, inline_slice, sample_slice):
    return {
        "crossline": np.asarray(label[crossline, inline_slice, sample_slice], dtype=np.float32),
        "inline": np.asarray(label[:, inline, sample_slice], dtype=np.float32),
        "sample": np.asarray(label[:, inline_slice, sample], dtype=np.float32),
    }


def extract_probability_planes(
    probability_sum,
    crossline_weight,
    weight_2d,
    crossline,
    inline,
    sample,
    inline_slice,
    sample_slice,
):
    eps = 1e-8
    cross_den = np.maximum(
        crossline_weight[crossline] * weight_2d[inline_slice, sample_slice], eps
    )
    inline_den = np.maximum(
        crossline_weight[:, None] * weight_2d[inline, sample_slice][None, :], eps
    )
    sample_den = np.maximum(
        crossline_weight[:, None] * weight_2d[inline_slice, sample][None, :], eps
    )
    return {
        "crossline": np.asarray(
            probability_sum[crossline, inline_slice, sample_slice] / cross_den,
            dtype=np.float32,
        ),
        "inline": np.asarray(
            probability_sum[:, inline, sample_slice] / inline_den, dtype=np.float32
        ),
        "sample": np.asarray(
            probability_sum[:, inline_slice, sample] / sample_den, dtype=np.float32
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", default="test4")
    parser.add_argument("--inline-start", type=int, default=400)
    parser.add_argument("--inline-stop", type=int, default=1400)
    parser.add_argument("--sample-start", type=int, default=700)
    parser.add_argument("--sample-stop", type=int, default=1300)
    parser.add_argument("--crossline", type=int, default=50)
    parser.add_argument("--inline", type=int, default=900)
    parser.add_argument("--sample", type=int, default=900)
    parser.add_argument("--crossline-exaggeration", type=float, default=4.0)
    parser.add_argument(
        "--coherence-run",
        type=Path,
        default=ROOT / "runs" / "dip_steered_coherence_baseline",
    )
    args = parser.parse_args()

    data_dir = ROOT / "processed_data" / "thebe_official" / "test" / args.block
    run_root = ROOT / "runs" / "thebe_final_test2_7"
    output_dir = run_root / "statistics" / "figures" / "three_dimensional"
    output_dir.mkdir(parents=True, exist_ok=True)

    amplitude = np.load(data_dir / "amplitude_norm.npy", mmap_mode="r")
    label = np.load(data_dir / "fault_label.npy", mmap_mode="r")
    inline_slice = slice(args.inline_start, args.inline_stop)
    sample_slice = slice(args.sample_start, args.sample_stop)
    cross_values = np.arange(amplitude.shape[0])
    inline_values = np.arange(args.inline_start, args.inline_stop)
    sample_values = np.arange(args.sample_start, args.sample_stop)

    amplitude_planes = extract_amplitude_planes(
        amplitude,
        args.crossline,
        args.inline,
        args.sample,
        inline_slice,
        sample_slice,
    )
    panels = [
        (
            "Expert interpretation",
            0.5,
            extract_label_planes(
                label,
                args.crossline,
                args.inline,
                args.sample,
                inline_slice,
                sample_slice,
            ),
        )
    ]
    coherence_path = args.coherence_run / args.block / "orthogonal_attribute_planes.npz"
    if coherence_path.exists():
        calibration_path = args.coherence_run / "calibration.json"
        coherence_threshold = 0.02
        if calibration_path.exists():
            coherence_threshold = float(
                json.loads(calibration_path.read_text(encoding="utf-8"))[
                    "selected_threshold"
                ]
            )
        coherence = np.load(coherence_path)
        panels.append(
            (
                "Locally dip-steered coherence",
                coherence_threshold,
                {
                    "crossline": np.asarray(
                        coherence["crossline"][inline_slice, sample_slice], dtype=np.float32
                    ),
                    "inline": np.asarray(
                        coherence["inline"][:, sample_slice], dtype=np.float32
                    ),
                    "sample": np.asarray(
                        coherence["sample"][:, inline_slice], dtype=np.float32
                    ),
                },
            )
        )
    crossline_weight, weight_2d = reconstruction_weights(amplitude.shape)
    for name, directory, threshold in MODELS:
        probability_sum = np.load(
            run_root / directory / args.block / "probability_sum.npy", mmap_mode="r"
        )
        panels.append(
            (
                name,
                threshold,
                extract_probability_planes(
                    probability_sum,
                    crossline_weight,
                    weight_2d,
                    args.crossline,
                    args.inline,
                    args.sample,
                    inline_slice,
                    sample_slice,
                ),
            )
        )

    plane_specs = {
        "crossline": (args.crossline, inline_values, sample_values),
        "inline": (args.inline, cross_values, sample_values),
        "sample": (args.sample, cross_values, inline_values),
    }
    pv.global_theme.background = "#FCFCFD"
    plotter = pv.Plotter(shape=(1, len(panels)), off_screen=True, window_size=(4000, 900))
    camera = None
    point_counts = {}
    for panel_index, (name, threshold, fault_planes) in enumerate(panels):
        plotter.subplot(0, panel_index)
        total_points = 0
        for axis in ("crossline", "inline", "sample"):
            fixed, a_values, b_values = plane_specs[axis]
            seismic = structured_plane(
                axis,
                fixed,
                a_values,
                b_values,
                amplitude_planes[axis],
                args.crossline_exaggeration,
            )
            plotter.add_mesh(
                seismic,
                scalars="amplitude",
                cmap="gray",
                clim=(-1.0, 1.0),
                show_scalar_bar=False,
                lighting=False,
            )
            cloud = fault_points(
                axis,
                fixed,
                a_values,
                b_values,
                fault_planes[axis],
                threshold,
                args.crossline_exaggeration,
            )
            total_points += cloud.n_points
            if cloud.n_points:
                if name == "Expert interpretation":
                    plotter.add_mesh(
                        cloud,
                        color="#F390CA",
                        point_size=4.0,
                        render_points_as_spheres=False,
                    )
                else:
                    plotter.add_mesh(
                        cloud,
                        scalars="fault_probability",
                        cmap="turbo",
                        clim=(threshold, 1.0),
                        point_size=4.0,
                        render_points_as_spheres=False,
                        show_scalar_bar=panel_index == len(panels) - 1 and axis == "sample",
                        scalar_bar_args={
                            "title": "Fault probability",
                            "vertical": False,
                            "position_x": 0.18,
                            "position_y": 0.04,
                            "width": 0.64,
                            "height": 0.07,
                            "title_font_size": 10,
                            "label_font_size": 8,
                        },
                    )
        point_counts[name] = total_points
        plotter.add_text(
            f"{name}\nthreshold = {threshold:.2f}" if name != "Expert interpretation" else name,
            position="upper_left",
            font_size=12,
            color="#1F2430",
        )
        plotter.add_bounding_box(color="#7A828F", line_width=1)
        plotter.show_axes()
        plotter.camera_position = [
            (1350, 1850, -180),
            (200, 900, -1000),
            (0, 0, 1),
        ]
        if camera is None:
            camera = plotter.camera_position
        else:
            plotter.camera_position = camera

    suffix = "expert_coherence_unet_hybrid_swin" if len(panels) == 5 else "expert_unet_hybrid_swin"
    screenshot = output_dir / f"{args.block}_orthogonal_{suffix}.png"
    plotter.screenshot(screenshot, transparent_background=False)
    plotter.close()

    metadata = {
        "block": args.block,
        "roi": {
            "crossline": [0, int(amplitude.shape[0])],
            "inline": [args.inline_start, args.inline_stop],
            "sample": [args.sample_start, args.sample_stop],
        },
        "orthogonal_slices": {
            "crossline": args.crossline,
            "inline": args.inline,
            "sample": args.sample,
        },
        "thresholds": {
            name: float(threshold)
            for name, threshold, _ in panels
            if name != "Expert interpretation"
        },
        "threshold_source": "Thebe val1-val2 only",
        "point_counts_on_three_planes": point_counts,
        "crossline_display_exaggeration": args.crossline_exaggeration,
        "prediction_postprocessing": "none",
        "figure": str(screenshot.relative_to(ROOT)),
    }
    metadata_path = screenshot.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
