import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticFaultDataset(Dataset):
    """Dataset for generated synthetic seismic fault samples.

    Expected sample layout:
        split/syn_xxxxx/amplitude.npy
        split/syn_xxxxx/fault_label.npy
        split/syn_xxxxx/metadata.json
    """

    def __init__(self, root, split="train", augment=False):
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.split_dir = self.root / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {self.split_dir}")

        self.samples = sorted(p for p in self.split_dir.iterdir() if p.is_dir())
        if not self.samples:
            raise ValueError(f"No samples found in {self.split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_dir = self.samples[index]
        amplitude = np.load(sample_dir / "amplitude.npy").astype(np.float32)
        label = np.load(sample_dir / "fault_label.npy").astype(np.float32)

        if self.augment:
            amplitude, label = self._augment(amplitude, label)

        # Shape: C, D, H, W for PyTorch Conv3d.
        amplitude = torch.from_numpy(amplitude[None, ...])
        label = torch.from_numpy(label[None, ...])

        metadata_path = sample_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        return {
            "amplitude": amplitude,
            "label": label,
            "sample_id": sample_dir.name,
            "metadata": metadata,
        }

    @staticmethod
    def _augment(amplitude, label):
        # Deterministic per-worker randomness is controlled by PyTorch worker seeds.
        if np.random.rand() < 0.5:
            amplitude = amplitude[::-1, :, :].copy()
            label = label[::-1, :, :].copy()
        if np.random.rand() < 0.5:
            amplitude = amplitude[:, ::-1, :].copy()
            label = label[:, ::-1, :].copy()
        if np.random.rand() < 0.5:
            amplitude = amplitude + np.random.normal(0.0, 0.02, size=amplitude.shape).astype(np.float32)
            amplitude = np.clip(amplitude, -1.0, 1.0)
        return amplitude, label


def load_manifest(root):
    manifest_path = Path(root) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))
