from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy import ndimage
from skimage.measure import block_reduce, marching_cubes
from skimage.morphology import skeletonize

from fault_experiments.visualize_thebe_3d import reconstruction_weights


ROOT = Path(__file__).resolve().parents[1]
THEBE_DATA = ROOT / "processed_data/thebe_official/test/test4"
THEBE_RUN = ROOT / "runs/thebe_final_test2_7"
THEBE_OUTPUT = THEBE_RUN / "statistics/figures/three_dimensional"
DELFT_DATA = ROOT / "processed_data/delft_external_center"
DELFT_RUN = ROOT / "runs/delft_frozen_external"
DELFT_OUTPUT = DELFT_RUN / "figures"

THEBE_MODELS = [
    ("U-Net", "unet3d", 0.50, "#A3BEFA"),
    ("Hybrid DSA", "dsa_hybrid_replay", 0.15, "#FFE15B"),
    ("SwinUNETR", "swin_unetr_f3chain", 0.40, "#A3D576"),
]
DELFT_MODELS = [
    ("U-Net", "unet", 0.50, "#A3BEFA"),
    ("Hybrid DSA", "hybrid_dsa", 0.15, "#FFE15B"),
    ("SwinUNETR", "swinunetr_f3chain", 0.40, "#A3D576"),
]

SURFACE = "#FCFCFD"
INK = "#1F2430"
MUTED = "#6F768A"
PINK = "#F390CA"


def reduce_binary(mask: np.ndarray, factors: tuple[int, int, int]) -> np.ndarray:
    return block_reduce(np.asarray(mask, dtype=np.uint8), block_size=factors, func=np.max).astype(bool)


def thin_section_traces(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    mask = np.asarray(mask, dtype=bool)
    thinned = np.zeros_like(mask)
    for section in range(mask.shape[0]):
        thinned[section] = skeletonize(mask[section])
    return thinned, {
        "voxels_before_thinning": int(mask.sum()),
        "voxels_after_thinning": int(thinned.sum()),
        "retained_fraction_after_thinning": float(thinned.sum() / max(mask.sum(), 1)),
        "method": "2D skeletonization independently on every first-axis seismic section",
    }


def retain_components(mask: np.ndarray, minimum_voxels: int) -> tuple[np.ndarray, dict]:
    structure = ndimage.generate_binary_structure(3, 3)
    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return mask, {
            "components_before": 0,
            "components_after": 0,
            "voxels_before": 0,
            "voxels_after": 0,
            "retained_fraction": 1.0,
        }
    sizes = np.bincount(labels.ravel())
    keep_ids = np.flatnonzero(sizes >= minimum_voxels)
    keep_ids = keep_ids[keep_ids != 0]
    cleaned = np.isin(labels, keep_ids)
    before = int(mask.sum())
    after = int(cleaned.sum())
    return cleaned, {
        "components_before": int(count),
        "components_after": int(len(keep_ids)),
        "voxels_before": before,
        "voxels_after": after,
        "retained_fraction": float(after / max(before, 1)),
        "minimum_component_voxels_on_display_grid": int(minimum_voxels),
    }


def binary_surface(
    mask: np.ndarray,
    factors: tuple[int, int, int],
    crossline_exaggeration: float,
    maximum_faces: int,
) -> tuple[pv.PolyData, dict]:
    if not mask.any():
        return pv.PolyData(), {"faces_before": 0, "faces_after": 0}
    spacing = np.asarray(
        [factors[0] * crossline_exaggeration, factors[1], factors[2]], dtype=np.float64
    )
    padded = np.pad(mask.astype(np.uint8), 1)
    vertices, faces, _, _ = marching_cubes(padded, level=0.5, spacing=tuple(spacing))
    vertices -= spacing
    points = np.column_stack((vertices[:, 1], vertices[:, 0], -vertices[:, 2]))
    vtk_faces = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).ravel()
    mesh = pv.PolyData(points, vtk_faces).clean()
    faces_before = int(mesh.n_cells)
    if mesh.n_cells > maximum_faces:
        reduction = 1.0 - maximum_faces / mesh.n_cells
        mesh = mesh.decimate_pro(reduction, preserve_topology=True).clean()
    faces_after = int(mesh.n_cells)
    return mesh, {
        "faces_before": faces_before,
        "faces_after": faces_after,
        "retained_face_fraction_after_decimation": float(faces_after / max(faces_before, 1)),
        "maximum_faces": int(maximum_faces),
        "preserve_topology": True,
    }


def seismic_context(
    amplitude: np.ndarray,
    factors: tuple[int, int, int],
    crossline_exaggeration: float,
) -> tuple[pv.StructuredGrid, pv.StructuredGrid]:
    ci = max(factors[0], 2)
    ii = max(factors[1], 4)
    ti = max(factors[2], 3)
    crosslines = np.arange(0, amplitude.shape[0], ci)
    inlines = np.arange(0, amplitude.shape[1], ii)
    samples = np.arange(0, amplitude.shape[2], ti)

    x_back, z_back = np.meshgrid(inlines, -samples, indexing="ij")
    y_back = np.zeros_like(x_back, dtype=np.float64)
    back = pv.StructuredGrid(x_back, y_back, z_back)
    back_values = np.asarray(amplitude[0, ::ii, ::ti], dtype=np.float32)
    back["amplitude"] = back_values.ravel(order="F")

    x_floor, y_floor = np.meshgrid(inlines, crosslines * crossline_exaggeration, indexing="ij")
    z_floor = np.full_like(x_floor, -(amplitude.shape[2] - 1), dtype=np.float64)
    floor = pv.StructuredGrid(x_floor, y_floor, z_floor)
    floor_values = np.asarray(amplitude[::ci, ::ii, -1], dtype=np.float32).T
    floor["amplitude"] = floor_values.ravel(order="F")
    return back, floor


def render_surface_panels(
    amplitude: np.ndarray,
    panels: list[tuple[str, np.ndarray, str, str]],
    output_path: Path,
    mesh_dir: Path,
    factors: tuple[int, int, int],
    minimum_component_voxels: int,
    crossline_exaggeration: float,
    maximum_faces: int,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    back, floor = seismic_context(amplitude, factors, crossline_exaggeration)
    plotter = pv.Plotter(
        shape=(1, len(panels)),
        off_screen=True,
        window_size=(800 * len(panels), 900),
        border=True,
    )
    plotter.set_background(SURFACE)
    metadata = {}
    camera = None
    for panel_index, (name, raw_mask, color, subtitle) in enumerate(panels):
        plotter.subplot(0, panel_index)
        thinned, thinning_stats = thin_section_traces(raw_mask)
        reduced = reduce_binary(thinned, factors)
        reduction_stats = {
            "voxels_before_block_reduction": int(thinned.sum()),
            "voxels_after_max_block_reduction": int(reduced.sum()),
            "block_factors": list(factors),
            "method": "binary maximum over each display-grid block",
        }
        cleaned, component_stats = retain_components(reduced, minimum_component_voxels)
        mesh, mesh_stats = binary_surface(
            cleaned,
            factors=factors,
            crossline_exaggeration=crossline_exaggeration,
            maximum_faces=maximum_faces,
        )
        safe_name = name.lower().replace(" ", "_").replace("-", "")
        mesh_path = mesh_dir / f"{safe_name}.vtp"
        if mesh.n_cells:
            mesh.save(mesh_path)
            plotter.add_mesh(
                mesh,
                color=color,
                opacity=0.52,
                smooth_shading=True,
                specular=0.18,
                ambient=0.28,
                diffuse=0.72,
                show_edges=False,
            )
        plotter.add_mesh(
            back,
            scalars="amplitude",
            cmap="gray",
            clim=(0, 1),
            opacity=0.24,
            show_scalar_bar=False,
            lighting=False,
        )
        plotter.add_mesh(
            floor,
            scalars="amplitude",
            cmap="gray",
            clim=(0, 1),
            opacity=0.20,
            show_scalar_bar=False,
            lighting=False,
        )
        plotter.add_bounding_box(color="#7A828F", line_width=1)
        plotter.add_text(name, position=(18, 840), font_size=13, color=INK)
        plotter.add_text(subtitle, position=(18, 795), font_size=9, color=MUTED)
        plotter.show_axes()
        plotter.camera_position = [
            (amplitude.shape[1] * 1.55, -amplitude.shape[0] * crossline_exaggeration * 1.35, 90),
            (amplitude.shape[1] * 0.47, amplitude.shape[0] * crossline_exaggeration * 0.48, -amplitude.shape[2] * 0.50),
            (0, 0, 1),
        ]
        if camera is None:
            camera = plotter.camera_position
        else:
            plotter.camera_position = camera
        plotter.camera.zoom(0.82)
        metadata[name] = {
            "display_grid_shape": list(cleaned.shape),
            "display_grid_factors": list(factors),
            "raw_voxel_count": int(np.asarray(raw_mask, dtype=bool).sum()),
            "raw_volume_fraction": float(np.asarray(raw_mask, dtype=bool).mean()),
            "display_thinning": thinning_stats,
            "display_grid_reduction": reduction_stats,
            "component_filter": component_stats,
            "mesh": mesh_stats,
            "mesh_file": str(mesh_path.relative_to(ROOT)) if mesh.n_cells else None,
        }
    plotter.screenshot(output_path, transparent_background=False)
    plotter.close()
    return metadata


def reconstruct_thebe_probability(
    model_directory: str,
    inline_slice: slice,
    sample_slice: slice,
    amplitude_shape: tuple[int, int, int],
) -> np.ndarray:
    probability_sum = np.load(
        THEBE_RUN / model_directory / "test4/probability_sum.npy", mmap_mode="r"
    )
    crossline_weight, weight_2d = reconstruction_weights(amplitude_shape)
    output = np.empty(
        (amplitude_shape[0], inline_slice.stop - inline_slice.start, sample_slice.stop - sample_slice.start),
        dtype=np.float32,
    )
    planar_weight = weight_2d[inline_slice, sample_slice]
    for crossline in range(amplitude_shape[0]):
        denominator = np.maximum(crossline_weight[crossline] * planar_weight, 1.0e-8)
        output[crossline] = probability_sum[crossline, inline_slice, sample_slice] / denominator
    return output


def make_thebe_figure(args) -> dict:
    amplitude_full = np.load(THEBE_DATA / "amplitude_norm.npy", mmap_mode="r")
    label_full = np.load(THEBE_DATA / "fault_label.npy", mmap_mode="r")
    inline_slice = slice(args.thebe_inline_start, args.thebe_inline_stop)
    sample_slice = slice(args.thebe_sample_start, args.thebe_sample_stop)
    amplitude = np.asarray(amplitude_full[:, inline_slice, sample_slice], dtype=np.float32)
    expert = np.asarray(label_full[:, inline_slice, sample_slice], dtype=bool)
    panels = [("Expert interpretation", expert, PINK, "Independent 3D fault labels")]
    for name, directory, threshold, color in THEBE_MODELS:
        probability = reconstruct_thebe_probability(
            directory, inline_slice, sample_slice, tuple(amplitude_full.shape)
        )
        summary = json.loads((THEBE_RUN / directory / "test4/summary.json").read_text(encoding="utf-8"))
        panels.append(
            (
                name,
                probability >= threshold,
                color,
                f"Frozen t={threshold:.2f} | full-block Dice={summary['dice']:.3f}",
            )
        )
    output_path = THEBE_OUTPUT / "test4_true_3d_fault_surfaces.png"
    panel_metadata = render_surface_panels(
        amplitude,
        panels,
        output_path,
        THEBE_OUTPUT / "test4_surface_meshes",
        factors=(1, 2, 2),
        minimum_component_voxels=6,
        crossline_exaggeration=8.0,
        maximum_faces=args.maximum_faces,
    )
    result = {
        "survey": "Thebe test4",
        "evidence_role": "expert-labeled three-dimensional field validation",
        "roi": {
            "crossline": [0, int(amplitude.shape[0])],
            "inline": [inline_slice.start, inline_slice.stop],
            "sample": [sample_slice.start, sample_slice.stop],
        },
        "rendering_only": "Frozen binary masks are thinned independently on each crossline section, max-pooled to the display grid, and components smaller than six display voxels are omitted. No metric uses this display processing.",
        "figure": str(output_path.relative_to(ROOT)),
        "panels": panel_metadata,
    }
    (THEBE_OUTPUT / "test4_true_3d_fault_surfaces.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def make_delft_figure(args) -> dict:
    amplitude_full = np.load(DELFT_DATA / "amplitude_norm.npy", mmap_mode="r")
    tfl_full = np.load(DELFT_DATA / "tfl_thinned.npy", mmap_mode="r")
    inline_slice = slice(96, 288)
    crossline_slice = slice(96, 288)
    sample_slice = slice(0, 128)
    selection = np.s_[inline_slice, crossline_slice, sample_slice]
    amplitude = np.asarray(amplitude_full[selection], dtype=np.float32)
    tfl = np.asarray(tfl_full[selection])
    panels = [("Traditional TFL", tfl >= 0.20, PINK, "Attribute comparator, not ground truth")]
    for name, directory, threshold, color in DELFT_MODELS:
        binary = np.load(DELFT_RUN / directory / "fault_binary.npy", mmap_mode="r")
        panels.append(
            (
                name,
                np.asarray(binary[selection], dtype=bool),
                color,
                f"Frozen Thebe threshold t={threshold:.2f}",
            )
        )
    output_path = DELFT_OUTPUT / "delft_true_3d_fault_surfaces.png"
    panel_metadata = render_surface_panels(
        amplitude,
        panels,
        output_path,
        DELFT_OUTPUT / "delft_surface_meshes",
        factors=(2, 2, 2),
        minimum_component_voxels=4,
        crossline_exaggeration=1.0,
        maximum_faces=args.maximum_faces,
    )
    result = {
        "survey": "Delft fixed centre ROI",
        "evidence_role": "label-free three-dimensional cross-survey stress test with TFL comparator",
        "display_subroi_within_frozen_inference_roi": {
            "inline_index": [inline_slice.start, inline_slice.stop],
            "crossline_index": [crossline_slice.start, crossline_slice.stop],
            "sample_index": [sample_slice.start, sample_slice.stop],
            "selection_policy": "geometric centre in both lateral directions; full time interval",
        },
        "tfl_threshold": 0.20,
        "rendering_only": "Frozen binary masks in the geometric-centre display sub-ROI are thinned independently on each inline section, max-pooled to the display grid, and components smaller than four display voxels are omitted. Full-ROI occupancy remains the reported quantitative descriptor; no accuracy is inferred from TFL and no metric uses this display processing.",
        "figure": str(output_path.relative_to(ROOT)),
        "panels": panel_metadata,
    }
    (DELFT_OUTPUT / "delft_true_3d_fault_surfaces.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", choices=("thebe", "delft", "both"), default="both")
    parser.add_argument("--thebe-inline-start", type=int, default=1400)
    parser.add_argument("--thebe-inline-stop", type=int, default=1800)
    parser.add_argument("--thebe-sample-start", type=int, default=700)
    parser.add_argument("--thebe-sample-stop", type=int, default=1100)
    parser.add_argument("--maximum-faces", type=int, default=180000)
    args = parser.parse_args()
    results = {}
    if args.survey in ("thebe", "both"):
        results["thebe"] = make_thebe_figure(args)
    if args.survey in ("delft", "both"):
        results["delft"] = make_delft_figure(args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
