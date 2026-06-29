import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from skimage.measure import block_reduce

from fault_experiments.infer_real_volume import blending_window, window_starts


ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("U-Net", "unet3d", 0.50, "#A3BEFA"),
    ("Hybrid DSA", "dsa_hybrid_replay", 0.15, "#FFE15B"),
]


def reconstruction_weights(shape, stride=64):
    pad_before = (128 - shape[0]) // 2
    pad_after = 128 - shape[0] - pad_before
    window = blending_window((128, 128, 128))[pad_before : 128 - pad_after]
    crossline_weight = window[:, 0, 0]
    planar = window[0] / max(float(crossline_weight[0]), 1e-8)
    weight_2d = np.zeros(shape[1:], dtype=np.float32)
    for inline_start in window_starts(shape[1], 128, stride):
        for sample_start in window_starts(shape[2], 128, stride):
            weight_2d[inline_start : inline_start + 128, sample_start : sample_start + 128] += planar
    return crossline_weight, weight_2d


def reduce_mask(mask, factors):
    return block_reduce(mask, block_size=factors, func=np.max).astype(bool)


def mask_points(mask, starts, factors, maximum_points, rng, crossline_exaggeration):
    coordinates = np.argwhere(mask)
    if len(coordinates) > maximum_points:
        coordinates = coordinates[rng.choice(len(coordinates), maximum_points, replace=False)]
    points = np.empty((len(coordinates), 3), dtype=np.float32)
    points[:, 0] = (starts[0] + coordinates[:, 0] * factors[0]) * crossline_exaggeration
    points[:, 1] = starts[1] + coordinates[:, 1] * factors[1]
    points[:, 2] = -(starts[2] + coordinates[:, 2] * factors[2])
    return points


def seismic_plane(amplitude, crossline, inline_slice, sample_slice, factors, crossline_exaggeration):
    image = np.asarray(
        amplitude[crossline, inline_slice, sample_slice], dtype=np.float32
    )[:: factors[1], :: factors[2]]
    inline = np.arange(inline_slice.start, inline_slice.stop, factors[1])[: image.shape[0]]
    sample = np.arange(sample_slice.start, sample_slice.stop, factors[2])[: image.shape[1]]
    yy, zz = np.meshgrid(inline, -sample, indexing="ij")
    xx = np.full_like(yy, crossline * crossline_exaggeration)
    grid = pv.StructuredGrid(xx, yy, zz)
    grid["amplitude"] = image.ravel(order="F")
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", default="test4")
    parser.add_argument("--inline-start", type=int, default=400)
    parser.add_argument("--inline-stop", type=int, default=1400)
    parser.add_argument("--sample-start", type=int, default=700)
    parser.add_argument("--sample-stop", type=int, default=1300)
    parser.add_argument("--max-points", type=int, default=80000)
    parser.add_argument("--crossline-exaggeration", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260627)
    args = parser.parse_args()

    data_dir = ROOT / "processed_data/thebe_official/test" / args.block
    run_root = ROOT / "runs/thebe_final_test2_7"
    output_dir = run_root / "statistics/figures/three_dimensional"
    output_dir.mkdir(parents=True, exist_ok=True)
    amplitude = np.load(data_dir / "amplitude_norm.npy", mmap_mode="r")
    label = np.load(data_dir / "fault_label.npy", mmap_mode="r")
    inline_slice = slice(args.inline_start, args.inline_stop)
    sample_slice = slice(args.sample_start, args.sample_stop)
    factors = (1, 4, 2)
    starts = (0, args.inline_start, args.sample_start)
    rng = np.random.default_rng(args.seed)

    expert_mask = reduce_mask(label[:, inline_slice, sample_slice], factors)
    masks = {"Expert interpretation": expert_mask}
    crossline_weight, weight_2d = reconstruction_weights(amplitude.shape)
    for name, directory, threshold, _ in MODELS:
        probability_sum = np.load(
            run_root / directory / args.block / "probability_sum.npy", mmap_mode="r"
        )
        reduced = np.zeros_like(expert_mask)
        for crossline in range(amplitude.shape[0]):
            denominator = np.maximum(
                crossline_weight[crossline] * weight_2d[inline_slice, sample_slice], 1e-8
            )
            prediction = probability_sum[crossline, inline_slice, sample_slice] / denominator >= threshold
            reduced[crossline] = reduce_mask(prediction, factors[1:])
        masks[name] = reduced

    pv.global_theme.background = "#FCFCFD"
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2100, 720))
    colors = {"Expert interpretation": "#F390CA", **{name: color for name, _, _, color in MODELS}}
    reference_crossline = 0
    plane = seismic_plane(
        amplitude,
        reference_crossline,
        inline_slice,
        sample_slice,
        factors,
        args.crossline_exaggeration,
    )
    point_counts = {}
    camera = None
    for panel, name in enumerate(("Expert interpretation", "U-Net", "Hybrid DSA")):
        plotter.subplot(0, panel)
        plotter.add_mesh(
            plane,
            scalars="amplitude",
            cmap="gray",
            clim=(-1, 1),
            opacity=0.24,
            show_scalar_bar=False,
        )
        points = mask_points(
            masks[name],
            starts,
            factors,
            args.max_points,
            rng,
            args.crossline_exaggeration,
        )
        point_counts[name] = {"available": int(masks[name].sum()), "rendered": len(points)}
        plotter.add_points(
            points,
            color=colors[name],
            point_size=2.1,
            render_points_as_spheres=False,
            opacity=0.82,
        )
        plotter.add_text(name, position="upper_left", font_size=12, color="#1F2430")
        plotter.add_bounding_box(color="#7A828F", line_width=1)
        plotter.show_axes()
        plotter.camera_position = [
            (900, 1700, -350),
            (190, 900, -1000),
            (0, 0, 1),
        ]
        if camera is None:
            camera = plotter.camera_position
        else:
            plotter.camera_position = camera
    screenshot = output_dir / f"{args.block}_expert_unet_hybrid_3d.png"
    plotter.screenshot(screenshot, transparent_background=False)
    plotter.close()

    metadata = {
        "block": args.block,
        "roi": {
            "crossline": [0, int(amplitude.shape[0])],
            "inline": [args.inline_start, args.inline_stop],
            "sample": [args.sample_start, args.sample_stop],
        },
        "block_reduction": factors,
        "maximum_rendered_points_per_panel": args.max_points,
        "seed": args.seed,
        "crossline_display_exaggeration": args.crossline_exaggeration,
        "seismic_reference_plane_crossline": reference_crossline,
        "point_counts": point_counts,
        "prediction_postprocessing": "none; block reduction retains any positive voxel",
        "figure": str(screenshot.relative_to(ROOT)),
    }
    (output_dir / f"{args.block}_3d_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
