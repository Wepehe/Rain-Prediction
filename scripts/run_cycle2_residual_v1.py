"""Run the authorized TRAIN/DEV-only Cycle-2 Residual V1 experiment."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter
from torch.nn import functional as F

from ontario_nowcast.models.pysteps_residual_unet import count_parameters
from ontario_nowcast.training.cycle2_residual import (
    CONFIG, DATA_ROOT, EXPECTED_FINAL_MANIFEST_SHA256, ROOT, ROWS,
    baseline_configuration, cache_system, canonical_hash, fit_normalization,
    guarded_rows, make_model, residual_losses, sha256_file,
    ResidualCacheDataset, EventBalancedBatchPlan,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def preflight() -> dict:
    rows = guarded_rows()
    final_manifest = DATA_ROOT / "final_manifest.json"
    final_hash = sha256_file(final_manifest)
    if final_hash != EXPECTED_FINAL_MANIFEST_SHA256:
        raise RuntimeError(f"replacement FINAL manifest hash mismatch: {final_hash}")
    counts = rows.groupby(["split", "row_type"]).size().to_dict()
    expected = {
        ("new_train", "active_precip"): 1728, ("new_train", "clean_initiation"): 512,
        ("new_train", "hard_negative"): 43, ("new_dev", "active_precip"): 288,
        ("new_dev", "clean_initiation"): 85, ("new_dev", "hard_negative"): 8,
    }
    if counts != expected:
        raise RuntimeError(f"frozen row-count mismatch: {counts}")
    model = make_model({"rate_log1p_mean": 0, "rate_log1p_std": 1,
                        "motion_mean": [0, 0], "motion_std": [1, 1]})
    params = count_parameters(model)
    residual_zero = float(model.residual_head.weight.abs().max()) == 0 and float(
        model.residual_head.bias.abs().max()) == 0
    if params != 3_060_440 or not residual_zero:
        raise RuntimeError(f"model freeze mismatch: params={params}, zero={residual_zero}")
    report = {
        "status": "passed", "sealed_final_accessed_for_scoring": False,
        "final_manifest_integrity_only_sha256": final_hash,
        "rows_manifest_sha256": sha256_file(ROWS), "config_sha256": sha256_file(CONFIG),
        "baseline_config": baseline_configuration(),
        "baseline_config_sha256": canonical_hash(baseline_configuration()),
        "counts": {f"{a}/{b}": int(v) for (a, b), v in counts.items()},
        "geometry": {"history": [10, 128, 128], "target": [20, 128, 128]},
        "model_parameters": params, "residual_head_exactly_zero": residual_zero,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                        "pysteps": importlib.metadata.version("pysteps"),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "repository_state": "initial worktree; no Git commit object exists",
        "code_hashes": {
            "runner": sha256_file(Path(__file__)),
            "training_module": sha256_file(Path("src/ontario_nowcast/training/cycle2_residual.py")),
            "model": sha256_file(Path("src/ontario_nowcast/models/pysteps_residual_unet.py")),
        },
    }
    atomic_json(ROOT / "preflight_report.json", report)
    return report


def cache(workers: int) -> None:
    preflight()
    rows = guarded_rows()
    tasks = [(system, group.copy()) for system, group in rows.groupby("system_id")]
    all_records = []
    completed = 0
    if workers == 1:
        iterator = ((system, cache_system(system, group)) for system, group in tasks)
        for system, records in iterator:
            all_records.extend(records); completed += 1
            print(f"CACHE {completed}/{len(tasks)} {system} rows={len(records)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(cache_system, system, group): system for system, group in tasks}
            for future in as_completed(futures):
                system = futures[future]; records = future.result()
                all_records.extend(records); completed += 1
                print(f"CACHE {completed}/{len(tasks)} {system} rows={len(records)}", flush=True)
    manifest = pd.DataFrame(all_records).sort_values(["split", "system_id", "row_id"])
    if len(manifest) != len(rows) or (manifest.split == "new_final").any():
        raise RuntimeError("cache manifest coverage or split guard failed")
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(ROOT / "cache_manifest.csv", index=False)
    normalization = fit_normalization(manifest)
    atomic_json(ROOT / "normalization.json", normalization)
    atomic_json(ROOT / "cache_status.json", {
        "status": "complete", "rows": len(manifest),
        "train_rows": int((manifest.split == "new_train").sum()),
        "dev_rows": int((manifest.split == "new_dev").sum()), "final_rows": 0,
        "fallback_counts": manifest.fallback.value_counts().to_dict(),
        "manifest_sha256": sha256_file(ROOT / "cache_manifest.csv"),
        "normalization_sha256": sha256_file(ROOT / "normalization.json"),
    })
    print("CACHE COMPLETE", flush=True)


def identity() -> None:
    manifest = pd.read_csv(ROOT / "cache_manifest.csv")
    norm = json.loads((ROOT / "normalization.json").read_text())
    model = make_model(norm).eval()
    row = manifest.loc[manifest.split == "new_train"].iloc[0]
    with np.load(row.cache_path) as z:
        h = torch.from_numpy(z["history"].astype(np.float32))[None]
        p = torch.from_numpy(z["pysteps"].astype(np.float32))[None]
        m = torch.from_numpy(z["motion"].astype(np.float32))[None]
        v = torch.from_numpy(z["validity"].astype(np.float32))[None]
    with torch.no_grad(): out = model(h, p, m, v)
    report = {
        "row_id": row.row_id,
        "max_abs_delta_log_rate": float(out["delta_log_rate"].abs().max()),
        "max_abs_corrected_minus_pysteps": float((out["corrected_rate"] - p).abs().max()),
        "output_shapes": {k: list(x.shape) for k, x in out.items()},
        "passed": bool(torch.equal(out["delta_log_rate"], torch.zeros_like(out["delta_log_rate"]))
                       and torch.allclose(out["corrected_rate"], p, atol=2e-5, rtol=2e-5)),
    }
    atomic_json(ROOT / "identity_check.json", report)
    if not report["passed"]: raise RuntimeError(f"identity check failed: {report}")
    print(json.dumps(report, indent=2))


def _batch(dataset, indices, device):
    items = [dataset[i] for i in indices]
    tensor_keys = ("history", "target", "validity", "target_validity", "pysteps", "motion")
    result = {k: torch.stack([x[k] for x in items]).to(device) for k in tensor_keys}
    result.update({k: [x[k] for x in items] for k in ("row_id", "system_id", "row_type")})
    return result


def _first_persistent(binary: np.ndarray) -> np.ndarray:
    persistent = binary[:-1] & binary[1:]
    found = persistent.any(axis=0)
    return np.where(found, (persistent.argmax(axis=0) + 1) * 6, np.nan)


def _row_metrics(truth, probability, rate, valid, threshold):
    mask = valid.astype(bool); obs = (truth > 0.1) & mask; pred = (probability >= threshold) & mask
    h = int((obs & pred).sum()); m = int((obs & ~pred).sum()); f = int((~obs & pred & mask).sum())
    ratio = lambda a, b: float(a / b) if b else float("nan")
    obs_fraction = uniform_filter(obs.astype(np.float32), size=(1, 19, 19))
    pred_fraction = uniform_filter(pred.astype(np.float32), size=(1, 19, 19))
    denom = np.square(obs_fraction).sum() + np.square(pred_fraction).sum()
    onset_obs = _first_persistent(obs); onset_pred = _first_persistent(pred)
    paired = np.isfinite(onset_obs) & np.isfinite(onset_pred)
    return {
        "hits": h, "misses": m, "false_alarms": f,
        "f1": ratio(2*h, 2*h+f+m), "fss18": float(1 - np.square(obs_fraction-pred_fraction).sum()/denom) if denom else 1.0,
        "onset_mae": float(np.mean(np.abs(onset_pred[paired]-onset_obs[paired]))) if paired.any() else np.nan,
        "brier": float(np.mean(np.square(probability[mask]-obs[mask]))),
        "rate_mae": float(np.mean(np.abs(rate[mask]-truth[mask]))),
    }


def evaluate_checkpoint(model, dataset, *, device, threshold=0.5, batch_size=4):
    model.eval(); rows = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = _batch(dataset, list(range(start, min(start+batch_size, len(dataset)))), device)
            out = model(batch["history"], batch["pysteps"], batch["motion"], batch["validity"])
            prob = torch.sigmoid(out["occurrence_logits"]).cpu().numpy()
            rate = out["corrected_rate"].cpu().numpy()
            for i in range(len(batch["row_id"])):
                met = _row_metrics(batch["target"][i].cpu().numpy(), prob[i], rate[i],
                                   batch["target_validity"][i].cpu().numpy(), threshold)
                met.update({k: batch[k][i] for k in ("row_id", "system_id", "row_type")})
                rows.append(met)
    frame = pd.DataFrame(rows)
    events = frame.groupby("system_id", as_index=False).mean(numeric_only=True)
    macro = events.mean(numeric_only=True).to_dict()
    return macro, frame, events


def train() -> None:
    preflight()
    manifest = pd.read_csv(ROOT / "cache_manifest.csv")
    if (manifest.split == "new_final").any(): raise PermissionError("FINAL cache entry found")
    norm = json.loads((ROOT / "normalization.json").read_text())
    train_rows = manifest.loc[manifest.split == "new_train"].reset_index(drop=True)
    dev_rows = manifest.loc[manifest.split == "new_dev"].reset_index(drop=True)
    train_data, dev_data = ResidualCacheDataset(train_rows), ResidualCacheDataset(dev_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(norm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    plan = EventBalancedBatchPlan(train_rows, batch_size=4, batches_per_epoch=571, seed=2718)
    history_path = ROOT / "training_history.csv"; history = []
    best_score = -float("inf"); best_epoch = 0; patience = 0
    status = {"training_state": "running", "device": str(device), "final_accessed": False}
    for epoch in range(1, 51):
        model.train(); batches, neg_audit = plan.epoch(epoch); sums = {k: 0.0 for k in
            ("train_total_loss", "occurrence_loss", "rate_loss", "spatial_loss", "residual_loss")}
        for indices in batches:
            batch = _batch(train_data, indices, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["history"], batch["pysteps"], batch["motion"], batch["validity"])
            loss, pieces = residual_losses(out, batch["target"], batch["target_validity"])
            if not torch.isfinite(loss):
                status.update({"training_state": "failed_nonfinite", "epoch": epoch}); atomic_json(ROOT/"training_status.json", status)
                raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad): raise FloatingPointError(f"nonfinite gradient at epoch {epoch}")
            optimizer.step(); sums["train_total_loss"] += float(loss.detach())
            for k, v in pieces.items(): sums[k] += float(v.detach())
        macro, _, _ = evaluate_checkpoint(model, dev_data, device=device, threshold=0.5)
        onset = macro.get("onset_mae", float("nan")); onset_for_score = onset if np.isfinite(onset) else 120.0
        score = macro["f1"] + 0.5 * macro["fss18"] - 0.002 * onset_for_score
        improved = score > best_score
        row = {"epoch": epoch, **{k: v/len(batches) for k,v in sums.items()},
               "dev_event_macro_f1": macro["f1"], "dev_event_macro_fss18": macro["fss18"],
               "dev_onset_mae": onset, "checkpoint_score": score,
               "learning_rate": optimizer.param_groups[0]["lr"], "best_checkpoint_changed": improved,
               **neg_audit}
        history.append(row); pd.DataFrame(history).to_csv(history_path, index=False)
        last = ROOT / "checkpoint_last.pt"
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "normalization": norm}, last)
        if improved:
            best_score=score; best_epoch=epoch; patience=0
            torch.save({"epoch": epoch, "model": model.state_dict(), "normalization": norm,
                        "checkpoint_score": score}, ROOT/"checkpoint_best.pt")
        else: patience += 1
        scheduler.step()
        status.update({"current_epoch": epoch, "best_epoch": best_epoch,
                       "best_checkpoint_score": best_score, "early_stopping_counter": patience,
                       "training_state": "running", "most_recent_checkpoint_sha256": sha256_file(last),
                       "hard_negative_audit": neg_audit})
        atomic_json(ROOT/"training_status.json", status)
        print(f"EPOCH {epoch} loss={row['train_total_loss']:.5f} dev_f1={macro['f1']:.4f} fss18={macro['fss18']:.4f} onset={onset:.2f} score={score:.4f} best={improved}", flush=True)
        if patience >= 8: break
    status.update({"training_state": "complete", "best_checkpoint_sha256": sha256_file(ROOT/"checkpoint_best.pt"),
                   "training_history_sha256": sha256_file(history_path)})
    atomic_json(ROOT/"training_status.json", status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["preflight", "cache", "identity", "train", "evaluate", "diagnostics"])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.phase == "preflight": print(json.dumps(preflight(), indent=2))
    elif args.phase == "cache": cache(args.workers)
    elif args.phase == "identity": identity()
    elif args.phase == "train": train()
    elif args.phase == "evaluate":
        from ontario_nowcast.training.cycle2_residual_eval import evaluate
        print(json.dumps(evaluate(), indent=2))
    else:
        from ontario_nowcast.training.cycle2_residual_diagnostics import run
        print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
