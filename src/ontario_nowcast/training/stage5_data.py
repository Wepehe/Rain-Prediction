"""Dataset utilities for frozen-A+ Stage 5 experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class Example:
    base_logits: torch.Tensor
    base_intensity: torch.Tensor
    base_expected: torch.Tensor
    hrrr: torch.Tensor
    goes: torch.Tensor
    target_occurrence: torch.Tensor
    target_intensity: torch.Tensor
    target_mask: torch.Tensor
    metadata: dict


class Dataset:
    def __init__(self, splits: set[str], crop: slice | None = None):
        table = pd.read_csv("artifacts/stage_5/cache/stage5_cache_manifest.csv")
        self.table = table[table.split.isin(splits)].reset_index(drop=True)
        self.crop = crop or slice(None)
        self.hn = json.loads(
            Path("artifacts/stage_5/gate/hrrr_apcp_normalization_train_only.json").read_text()
        )
        self.gn = json.loads(
            Path("artifacts/stage_4b/b2_c13_cooling/training_metadata.json").read_text()
        )["normalization"]["goes"]

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        cache = np.load(row.stage5_cache_path)
        source = np.load(row.radar_goes_tensor_path)
        c = self.crop
        hrrr_raw = cache["hrrr_apcp_hourly"][:, c, c]
        hrrr_value = np.nan_to_num((np.log1p(hrrr_raw) - self.hn["mean"]) / self.hn["std"])
        hrrr = np.stack([hrrr_value, np.isfinite(hrrr_raw).astype(np.float32)], 1).astype(
            np.float32
        )
        indices = self.gn["goes_channel_indices"]
        goes_raw = source["goes"][indices, :, c, c]
        goes_mask = source["goes_valid_mask"][indices, :, c, c].astype(np.float32)
        mean = np.asarray(self.gn["mean"])[:, None, None, None]
        std = np.asarray(self.gn["std"])[:, None, None, None]
        goes_value = (np.nan_to_num(goes_raw) - mean) / std
        goes = np.concatenate(
            [np.moveaxis(goes_value, 1, 0), np.moveaxis(goes_mask, 1, 0)], 1
        ).astype(np.float32)
        target = source["target_rate_mm_hr"][:, c, c].astype(np.float32)
        mask = source["target_valid_mask"][:, c, c].astype(np.float32)
        thresholds = np.asarray([0.1, 1.0, 2.5, 5.0])[None, :, None, None]
        return Example(
            torch.from_numpy(cache["aplus_logits"][:, :, c, c]),
            torch.from_numpy(cache["aplus_intensity_raw"][:, c, c]),
            torch.from_numpy(cache["aplus_expected_rate_mm_hr"][:, c, c]),
            torch.from_numpy(hrrr),
            torch.from_numpy(goes),
            torch.from_numpy(
                ((target[:, None] >= thresholds) & (mask[:, None] > 0)).astype(np.float32)
            ),
            torch.from_numpy(np.log1p(np.nan_to_num(target))),
            torch.from_numpy(mask),
            row.to_dict(),
        )


def collate(items):
    names = (
        "base_logits",
        "base_intensity",
        "base_expected",
        "hrrr",
        "goes",
        "target_occurrence",
        "target_intensity",
        "target_mask",
    )
    return {
        **{name: torch.stack([getattr(item, name) for item in items]) for name in names},
        "metadata": [item.metadata for item in items],
    }
