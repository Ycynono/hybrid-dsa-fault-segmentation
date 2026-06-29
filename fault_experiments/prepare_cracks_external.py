import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external_data/CRACKS/extracted"
OUTPUT = ROOT / "processed_data/cracks_external_v1"
MISSING_SECTIONS = [9, 185, 249, 336]
FAULT_COLORS = {
    "certain": (31, 119, 180),
    "uncertain": (44, 160, 44),
}


def packed_rgb(array):
    array = array.astype(np.uint32)
    return (array[..., 0] << 16) | (array[..., 1] << 8) | array[..., 2]


def decode_seismic(path, palette_lookup):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    unique, inverse = np.unique(packed_rgb(rgb), return_inverse=True)
    try:
        indices = np.array([palette_lookup[int(value)] for value in unique], dtype=np.uint8)
    except KeyError as exc:
        raise ValueError(f"Unexpected seismic color {exc.args[0]} in {path}") from exc
    decoded = indices[inverse].reshape(rgb.shape[:2]).astype(np.float32) / 255.0
    return decoded.T


def decode_expert(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    certain = np.all(rgb == FAULT_COLORS["certain"], axis=-1)
    uncertain = np.all(rgb == FAULT_COLORS["uncertain"], axis=-1)
    return (certain | uncertain).T, certain.T, uncertain.T


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    palette = (
        plt.get_cmap("seismic")(np.linspace(0.0, 1.0, 256))[:, :3] * 255
    ).astype(np.uint8)
    palette_lookup = {
        int(packed_rgb(color[None, :])[0]): index for index, color in enumerate(palette)
    }
    amplitude = np.empty((400, 701, 255), dtype=np.float16)
    available = []
    for section in range(1, 401):
        path = SOURCE / "images" / f"section_{section:03d}.png"
        if path.exists():
            amplitude[section - 1] = decode_seismic(path, palette_lookup).astype(np.float16)
            available.append(section)
    for section in MISSING_SECTIONS:
        lower = max(value for value in available if value < section)
        upper = min(value for value in available if value > section)
        weight = (section - lower) / (upper - lower)
        amplitude[section - 1] = (
            (1.0 - weight) * amplitude[lower - 1].astype(np.float32)
            + weight * amplitude[upper - 1].astype(np.float32)
        ).astype(np.float16)
    np.save(OUTPUT / "amplitude_01.npy", amplitude)

    expert_files = sorted((SOURCE / "Fault segmentations/expert").glob("section_*.png"))
    expert_sections = [int(path.stem.split("_")[1]) for path in expert_files]
    audit_sections = expert_sections[::2]
    reserve_sections = expert_sections[1::2]
    masks = []
    certain_masks = []
    uncertain_masks = []
    for path in expert_files:
        mask, certain, uncertain = decode_expert(path)
        masks.append(mask)
        certain_masks.append(certain)
        uncertain_masks.append(uncertain)
    masks = np.stack(masks)
    certain_masks = np.stack(certain_masks)
    uncertain_masks = np.stack(uncertain_masks)
    audit_positions = [expert_sections.index(section) for section in audit_sections]
    reserve_positions = [expert_sections.index(section) for section in reserve_sections]
    np.save(OUTPUT / "audit_expert_fault_masks.npy", masks[audit_positions])
    np.save(OUTPUT / "audit_expert_certain_masks.npy", certain_masks[audit_positions])
    np.save(OUTPUT / "audit_expert_uncertain_masks.npy", uncertain_masks[audit_positions])
    reserve_dir = OUTPUT / "sealed_reserve"
    reserve_dir.mkdir(exist_ok=True)
    np.save(reserve_dir / "reserve_expert_fault_masks.npy", masks[reserve_positions])
    np.save(reserve_dir / "reserve_expert_certain_masks.npy", certain_masks[reserve_positions])
    np.save(reserve_dir / "reserve_expert_uncertain_masks.npy", uncertain_masks[reserve_positions])

    split = {
        "policy_created_before_model_inference": True,
        "policy": "sorted expert sections alternated; even positions audit, odd positions sealed reserve",
        "expert_sections": expert_sections,
        "audit_sections": audit_sections,
        "reserve_sections": reserve_sections,
        "reserve_access_policy": "do not load during development or audit evaluation",
    }
    (OUTPUT / "expert_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    metadata = {
        "dataset": "CRACKS v2",
        "doi": "10.5281/zenodo.13926822",
        "license": "CC BY 4.0",
        "source_shape": [396, 255, 701],
        "prepared_shape": list(amplitude.shape),
        "prepared_axis_order": ["section", "trace", "depth"],
        "normalization": "exact inverse lookup of the 256-color matplotlib seismic palette, mapped to [0,1]",
        "missing_sections": MISSING_SECTIONS,
        "missing_section_policy": "linear interpolation between adjacent available sections",
        "expert_label_policy": "green uncertain and blue certain are faults; orange no-fault and white unlabeled are not faults",
        "expert_section_count": len(expert_sections),
        "audit_section_count": len(audit_sections),
        "sealed_reserve_section_count": len(reserve_sections),
        "audit_fault_pixels": int(masks[audit_positions].sum()),
        "reserve_fault_pixels": int(masks[reserve_positions].sum()),
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"split": split, "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
