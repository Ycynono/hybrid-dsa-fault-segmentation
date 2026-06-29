import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fault_experiments.infer_real_volume import window_starts


class ThebeBalancedPatchDataset(Dataset):
    def __init__(self, root, split, samples_per_epoch=200, augment=False, seed=20261101):
        self.root = Path(root)
        self.split = split
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 1000))
        self.blocks = []
        self.positive = []
        self.negative = []
        split_dir = self.root / split
        for block_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            amplitude = np.load(block_dir / "amplitude_norm.npy", mmap_mode="r")
            label = np.load(block_dir / "fault_label.npy", mmap_mode="r")
            if amplitude.shape != label.shape:
                raise ValueError(f"Shape mismatch in {block_dir}: {amplitude.shape}/{label.shape}")
            block_index = len(self.blocks)
            self.blocks.append({"id": block_dir.name, "amplitude": amplitude, "label": label})
            projection = np.any(label, axis=0)
            for inline_start in window_starts(label.shape[1], 128, 64):
                for sample_start in window_starts(label.shape[2], 128, 64):
                    item = (block_index, inline_start, sample_start)
                    if projection[
                        inline_start : inline_start + 128,
                        sample_start : sample_start + 128,
                    ].any():
                        self.positive.append(item)
                    else:
                        self.negative.append(item)
        if not self.positive or not self.negative:
            raise ValueError(f"Thebe {split} requires both positive and negative patches")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        pool = self.positive if index % 2 == 0 else self.negative
        if self.augment:
            block_index, inline_start, sample_start = pool[self.rng.integers(0, len(pool))]
        else:
            per_class = max((self.samples_per_epoch + 1) // 2, 1)
            pool_index = min((index // 2) * len(pool) // per_class, len(pool) - 1)
            block_index, inline_start, sample_start = pool[pool_index]
        block = self.blocks[block_index]
        amplitude = np.asarray(
            block["amplitude"][:, inline_start : inline_start + 128, sample_start : sample_start + 128],
            dtype=np.float32,
        )
        label = np.asarray(
            block["label"][:, inline_start : inline_start + 128, sample_start : sample_start + 128],
            dtype=np.float32,
        )
        pad_before = (128 - amplitude.shape[0]) // 2
        pad_after = 128 - amplitude.shape[0] - pad_before
        amplitude = np.pad(amplitude, ((pad_before, pad_after), (0, 0), (0, 0)), mode="reflect")
        target = np.pad(label, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant")
        valid = np.zeros_like(target, dtype=bool)
        valid[pad_before : 128 - pad_after] = True
        if self.augment and self.rng.random() < 0.5:
            amplitude = amplitude[::-1].copy()
            target = target[::-1].copy()
            valid = valid[::-1].copy()
        if self.augment and self.rng.random() < 0.5:
            amplitude = amplitude[:, ::-1].copy()
            target = target[:, ::-1].copy()
            valid = valid[:, ::-1].copy()
        return {
            "amplitude": torch.from_numpy(np.ascontiguousarray(amplitude[None])),
            "target": torch.from_numpy(np.ascontiguousarray(target[None])),
            "valid": torch.from_numpy(np.ascontiguousarray(valid[None])),
            "block_id": block["id"],
        }

    def summary(self):
        return {
            "split": self.split,
            "block_count": len(self.blocks),
            "positive_candidate_count": len(self.positive),
            "negative_candidate_count": len(self.negative),
            "samples_per_epoch": self.samples_per_epoch,
        }
