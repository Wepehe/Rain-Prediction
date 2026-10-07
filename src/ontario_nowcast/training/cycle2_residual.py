"""Guarded preparation and training utilities for Cycle-2 Residual V1.

All public entry points in this module accept only the combined TRAIN/DEV row
manifest.  They deliberately have no argument through which the sealed FINAL
manifest can be supplied.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from ..models.pysteps_residual_unet import PySTEPSResidualUNetV1
from ..sample_evaluation import _downsample_max

ROOT = Path("artifacts/cycle2/residual_v1")
DATA_ROOT = Path("artifacts/cycle2/data")
ROWS = DATA_ROOT / "train_dev_rows.csv"
CONFIG = Path("configs/cycle2_residual_v1.yaml")
EXPECTED_FINAL_MANIFEST_SHA256 = "d71dd81da4663c98c84858a6890ee824922033b68140fb38cedaea7928c9b0c3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guarded_rows() -> pd.DataFrame:
    rows = pd.read_csv(ROWS)
    expected = {"new_train", "new_dev"}
    actual = set(rows["split"].unique())
    if not actual <= expected or "new_final" in actual:
        raise PermissionError(f"unauthorized split in TRAIN/DEV manifest: {sorted(actual)}")
    if rows["row_id"].duplicated().any():
        raise ValueError("duplicate row IDs in TRAIN/DEV manifest")
    return rows


def baseline_configuration() -> dict:
    return {
        "method": "pysteps deterministic extrapolation",
        "motion_method": "LK",
        "motion_input_frames": 3,
        "history_frames": 10,
        "lead_steps": 20,
        "timestep_minutes": 6,
        "transform": "dB",
        "rain_threshold_mm_hr": 0.1,
        "zerovalue_db": -15.0,
        "fallback": "zero velocity plus latest-field extrapolation only if LK fails",
    }


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def deterministic_pysteps_with_motion(history: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Reproduce the validated baseline and also return its LK velocity."""
    from pysteps import motion, nowcasts
    from pysteps.utils import transformation

    finite = np.nan_to_num(np.asarray(history, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    transformed, metadata = transformation.dB_transform(
        finite,
        metadata={"unit": "mm/h", "transform": None},
        threshold=0.1,
        zerovalue=-15.0,
    )
    transformed[~np.isfinite(transformed)] = metadata.get("zerovalue", -15.0)
    fallback = "none"
    try:
        velocity = np.asarray(motion.get_method("LK")(transformed[-3:]), np.float32)
        if velocity.shape != (2, 128, 128) or not np.isfinite(velocity).all():
            raise ValueError("invalid LK velocity")
    except Exception as exc:  # dry scenes may contain no trackable features
        velocity = np.zeros((2, 128, 128), np.float32)
        fallback = f"zero_velocity:{type(exc).__name__}"
    with contextlib.redirect_stdout(io.StringIO()):
        forecast = nowcasts.get_method("extrapolation")(transformed[-1], velocity, 20)
    result, _ = transformation.dB_transform(forecast, metadata=metadata, inverse=True)
    result = np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    return np.asarray(result, np.float32), velocity, fallback


def _extract_from_loaded(row, rates: np.ndarray, times: pd.DatetimeIndex):
    issue = pd.Timestamp(row.issue_time_utc)
    anchor = int(times.get_indexer([issue])[0])
    if anchor < 9 or anchor + 20 >= len(times):
        raise ValueError(f"unsupported time window for {row.row_id}")
    y0, y1, x0, x1 = (int(getattr(row, k)) for k in ("y0", "y1", "x0", "x1"))
    tile = rates[:, y0 : y1 + 1, x0 : x1 + 1]
    history, target = tile[anchor - 9 : anchor + 1], tile[anchor + 1 : anchor + 21]
    if history.shape != (10, 128, 128) or target.shape != (20, 128, 128):
        raise ValueError(f"bad tensor shape for {row.row_id}: {history.shape}, {target.shape}")
    validity = np.isfinite(history)
    target_validity = np.isfinite(target)
    return history, target, validity, target_validity


def cache_system(system_id: str, rows: pd.DataFrame, *, overwrite: bool = False) -> list[dict]:
    """Cache one system; intended for resumable sequential or process-pool use."""
    if (rows["split"] == "new_final").any():
        raise PermissionError("NEW FINAL cache generation is forbidden")
    source_paths = rows["source_path"].unique()
    if len(source_paths) != 1:
        raise ValueError(f"{system_id} has {len(source_paths)} source paths")
    source = Path(source_paths[0])
    expected_source_hashes = rows["source_sha256"].unique()
    if len(expected_source_hashes) != 1 or sha256_file(source) != expected_source_hashes[0]:
        raise ValueError(f"source hash mismatch for {system_id}")
    with np.load(source) as data:
        times = pd.to_datetime(data["times"], utc=True)
        rates = _downsample_max(data["rate_mm_hr"], 2)
    config_hash = canonical_hash(baseline_configuration())
    records = []
    for row in rows.itertuples(index=False):
        out = ROOT / "cache" / row.split / f"{row.row_id}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        fallback = "cached"
        cache_current = False
        if out.exists() and not overwrite:
            with np.load(out) as prior:
                cache_current = "target_validity" in prior.files
        if overwrite or not cache_current:
            history, target, validity, target_validity = _extract_from_loaded(row, rates, times)
            forecast, velocity, fallback = deterministic_pysteps_with_motion(history)
            # Float16 keeps the detached cache tractable; hashes cover serialized bytes.
            np.savez_compressed(
                out,
                history=np.nan_to_num(history, nan=0.0).astype(np.float16),
                target=np.nan_to_num(target, nan=0.0).astype(np.float16),
                validity=validity.astype(np.uint8),
                target_validity=target_validity.astype(np.uint8),
                pysteps=forecast.astype(np.float16),
                motion=velocity.astype(np.float16),
            )
        with np.load(out) as cached:
            shapes = {k: list(cached[k].shape) for k in cached.files}
        required = {"history": [10, 128, 128], "target": [20, 128, 128],
                    "validity": [10, 128, 128], "target_validity": [20, 128, 128],
                    "pysteps": [20, 128, 128],
                    "motion": [2, 128, 128]}
        if shapes != required:
            raise ValueError(f"cache integrity failure for {row.row_id}: {shapes}")
        records.append({
            "row_id": row.row_id, "system_id": system_id, "split": row.split,
            "row_type": row.row_type, "issue_time_utc": row.issue_time_utc,
            "source_path": str(source), "source_sha256": expected_source_hashes[0],
            "pysteps_config_sha256": config_hash, "cache_path": str(out),
            "output_sha256": sha256_file(out), "fallback": fallback,
        })
    return records


def fit_normalization(cache_manifest: pd.DataFrame) -> dict:
    train = cache_manifest.loc[cache_manifest["split"] == "new_train"]
    if len(train) == 0 or (cache_manifest["split"] == "new_final").any():
        raise PermissionError("normalization requires nonempty TRAIN and forbids FINAL")
    rate_n = 0
    rate_s = rate_ss = 0.0
    motion_n = np.zeros(2, np.int64)
    motion_s = np.zeros(2, np.float64)
    motion_ss = np.zeros(2, np.float64)
    for row in train.itertuples(index=False):
        with np.load(row.cache_path) as z:
            valid = z["validity"].astype(bool)
            history = np.log1p(z["history"].astype(np.float32)[valid]).astype(np.float64)
            # Include detached baseline rates because the same rate normalization serves both inputs.
            baseline = np.log1p(z["pysteps"].astype(np.float32)).astype(np.float64).ravel()
            rate = np.concatenate((history, baseline))
            rate_n += rate.size; rate_s += rate.sum(); rate_ss += np.square(rate).sum()
            motion = z["motion"].astype(np.float64)
            for c in range(2):
                vals = motion[c].ravel(); motion_n[c] += vals.size
                motion_s[c] += vals.sum(); motion_ss[c] += np.square(vals).sum()
    rate_mean = rate_s / rate_n
    rate_std = math.sqrt(max(rate_ss / rate_n - rate_mean**2, 1e-12))
    motion_mean = motion_s / motion_n
    motion_std = np.sqrt(np.maximum(motion_ss / motion_n - motion_mean**2, 1e-12))
    result = {
        "source_split": "new_train", "dev_used": False, "final_used": False,
        "rate_log1p_mean": float(rate_mean), "rate_log1p_std": float(rate_std),
        "motion_mean": motion_mean.tolist(), "motion_std": motion_std.tolist(),
        "rate_values": int(rate_n), "motion_values_per_channel": motion_n.tolist(),
        "training_rows": int(len(train)),
    }
    result["statistics_sha256"] = canonical_hash(result)
    return result


class ResidualCacheDataset(Dataset):
    def __init__(self, rows: pd.DataFrame):
        if (rows["split"] == "new_final").any():
            raise PermissionError("NEW FINAL cannot enter a residual dataset")
        self.rows = rows.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with np.load(row.cache_path) as z:
            item = {k: torch.from_numpy(z[k].astype(np.float32)) for k in
                    ("history", "target", "validity", "target_validity", "pysteps", "motion")}
        item.update({"row_id": row.row_id, "system_id": row.system_id,
                     "row_type": row.row_type})
        return item


class EventBalancedBatchPlan:
    """Deterministic per-epoch plan with one negative in ~25% of batches."""
    def __init__(self, rows: pd.DataFrame, batch_size: int, batches_per_epoch: int, seed: int = 2718):
        if batch_size < 2:
            raise ValueError("batch size must be at least two")
        self.rows, self.batch_size, self.batches = rows.reset_index(drop=True), batch_size, batches_per_epoch
        self.seed = seed
        self.positive = defaultdict(list); self.negative = []
        for i, r in self.rows.iterrows():
            (self.negative if r.row_type == "hard_negative" else self.positive[r.system_id]).append(i)
        self.systems = sorted(self.positive)

    def epoch(self, epoch: int) -> tuple[list[list[int]], dict]:
        rng = np.random.default_rng(self.seed + epoch)
        neg = np.asarray(self.negative); rng.shuffle(neg); neg_cursor = 0
        counts = defaultdict(int); result = []; positive_cursor = epoch % len(self.systems)
        for b in range(self.batches):
            use_negative = ((b + epoch) % 4 == 0) and len(neg)
            batch = []
            if use_negative:
                if neg_cursor and neg_cursor % len(neg) == 0: rng.shuffle(neg)
                idx = int(neg[neg_cursor % len(neg)]); neg_cursor += 1
                batch.append(idx); counts[idx] += 1
            while len(batch) < self.batch_size:
                system = self.systems[positive_cursor % len(self.systems)]
                positive_cursor += 1
                candidates = self.positive[system]
                # Active rows receive 3x the draw weight of clean initiation rows.
                active = [i for i in candidates if self.rows.iloc[i].row_type == "active_precip"]
                clean = [i for i in candidates if self.rows.iloc[i].row_type == "clean_initiation"]
                pool = active if active and (len(batch) % 4 != 3 or not clean) else clean
                idx = int(rng.choice(pool)); batch.append(idx)
            result.append(batch)
        drawn = [counts[i] for i in self.negative]
        return result, {"hard_negative_draws": int(sum(drawn)),
                        "unique_hard_negatives": int(sum(x > 0 for x in drawn)),
                        "max_hard_negative_repeat": int(max(drawn, default=0))}


def focal_loss(logits, target, valid, alpha=0.25, gamma=2.0):
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = target * p + (1 - target) * (1 - p)
    weight = target * alpha + (1 - target) * (1 - alpha)
    values = weight * (1 - pt).pow(gamma) * ce * valid
    return values.sum() / valid.sum().clamp_min(1)


def residual_losses(output, target, valid):
    wet = (target > 0.1).float(); valid = valid.float()
    occurrence = focal_loss(output["occurrence_logits"], wet, valid)
    truth_log = torch.log1p(torch.clamp(target, min=0))
    huber = F.smooth_l1_loss(output["corrected_log_rate"], truth_log, reduction="none")
    rate_weight = wet + 0.05 * (1 - wet)
    rate = (huber * rate_weight * valid).sum() / (rate_weight * valid).sum().clamp_min(1)
    probability = torch.sigmoid(output["occurrence_logits"])
    # 18-km neighborhood at 2 km/pixel -> predeclared odd 9-pixel pooling kernel.
    p_pool = F.avg_pool2d(probability, 9, stride=1, padding=4)
    y_pool = F.avg_pool2d(wet, 9, stride=1, padding=4)
    spatial = 1 - (2 * (p_pool * y_pool).sum() + 1e-6) / (
        p_pool.square().sum() + y_pool.square().sum() + 1e-6)
    residual = (output["delta_log_rate"].abs() * valid).sum() / valid.sum().clamp_min(1)
    total = occurrence + 0.5 * rate + 0.25 * spatial + 0.05 * residual
    return total, {"occurrence_loss": occurrence, "rate_loss": rate,
                   "spatial_loss": spatial, "residual_loss": residual}


def make_model(normalization: dict) -> PySTEPSResidualUNetV1:
    model = PySTEPSResidualUNetV1()
    model.set_normalization(rate_mean=normalization["rate_log1p_mean"],
                            rate_std=normalization["rate_log1p_std"],
                            motion_mean=normalization["motion_mean"],
                            motion_std=normalization["motion_std"])
    return model
