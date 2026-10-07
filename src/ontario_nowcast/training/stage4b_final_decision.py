"""Final, predeclared DEV-only Stage 4B decision between A+ and B2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ..evaluation.metrics import brier_score, categorical_metrics
from ..evaluation.onset import first_crossing_minutes, onset_metrics
from ..models.radar_convlstm import build_radar_model, expected_rate_mm_hr
from ..models.radar_goes import build_stage4b_model
from .stage4b_lifecycle import _goes_inputs, _radar_inputs
from .stage4b_train import _load_npz

GRID = tuple(round(x, 2) for x in np.arange(0.05, 1.0, 0.05))
RAIN = 0.1
LEADS = (10, 15)  # 60 and 90 minutes
SEED = 4317


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lifecycle(obs: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    observed = first_crossing_minutes(obs, RAIN, 6)
    predicted = first_crossing_minutes(prob, threshold, 6)
    metrics = onset_metrics(observed.ravel(), predicted.ravel())
    hit = int(metrics["paired_events"])
    false = int(metrics["false_events"])
    miss = int(metrics["missed_events"])
    precision = hit / max(hit + false, 1)
    recall = hit / max(hit + miss, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    event_prob = np.nanmax(prob, axis=0)
    event_obs = np.isfinite(observed)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "csi": hit / max(hit + false + miss, 1),
        "far": false / max(hit + false, 1),
        "onset_mae_minutes": float(metrics["mae_minutes"]),
        "onset_bias_minutes": float(metrics["median_error_minutes"]),
        "false_initiation_fraction": false / max(hit + false, 1),
        "brier_score": brier_score(event_obs.astype(float), event_prob),
    }


def _models(device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    ap_cfg = _yaml(Path("configs/experiments/stage_4b_a_plus_radar_capacity.yaml"))
    b2_cfg = _yaml(Path("configs/experiments/stage_4b_b2_c13_cooling.yaml"))
    ap_meta = json.loads(Path("artifacts/stage_4b/a_plus_radar/training_metadata.json").read_text())
    b2_meta = json.loads(Path("artifacts/stage_4b/b2_c13_cooling/training_metadata.json").read_text())
    ap = build_radar_model({"model": ap_cfg["model"], "data": ap_cfg["data"]}).to(device)
    b2 = build_stage4b_model(b2_cfg).to(device)
    ap.load_state_dict(torch.load("artifacts/stage_4b/a_plus_radar/model_best.pt", map_location=device))
    b2.load_state_dict(torch.load("artifacts/stage_4b/b2_c13_cooling/model_best.pt", map_location=device))
    return {"A_plus": ap.eval(), "B2": b2.eval()}, {"A_plus": ap_meta, "B2": b2_meta, "b2_cfg": b2_cfg}


def _predict_dev(output: Path) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    manifest = pd.read_csv("artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv")
    dev = manifest[manifest.split.astype(str).isin(["dev", "stage4_dev_repair"])].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, meta = _models(device)
    cache: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    # Event-local cyclic shuffle preserves season/regime and avoids batch-order artifacts.
    grouped = list(dev.groupby("event_id", sort=True))
    for group_index, (event_id, event_rows) in enumerate(grouped):
        payloads = [_load_npz(Path(str(r.tensor_path))) for r in event_rows.itertuples(index=False)]
        radar = torch.from_numpy(np.stack([_radar_inputs(p, meta["B2"]["normalization"]) for p in payloads])).to(device)
        goes = torch.from_numpy(np.stack([_goes_inputs(p, meta["B2"]["normalization"], channel_indices=[0,1,2,3,4]) for p in payloads])).to(device)
        other_rows = grouped[(group_index + 1) % len(grouped)][1]
        other_payloads = [_load_npz(Path(str(r.tensor_path))) for r in other_rows.itertuples(index=False)]
        global_goes = torch.from_numpy(
            np.stack(
                [
                    _goes_inputs(
                        other_payloads[j % len(other_payloads)],
                        meta["B2"]["normalization"],
                        channel_indices=[0, 1, 2, 3, 4],
                    )
                    for j in range(len(payloads))
                ]
            )
        ).to(device)
        variants = {
            "normal": goes,
            "global_shuffle": global_goes,
            "similar_shuffle": torch.roll(goes, 1, 0),
            "all_goes_zero": torch.zeros_like(goes),
            "cooling_zero": goes.clone(),
            "raw_c13_zero": goes.clone(),
        }
        variants["cooling_zero"][:, :, 1:5] = 0
        variants["raw_c13_zero"][:, :, 0] = 0
        with torch.no_grad():
            ap_l, ap_i = models["A_plus"](radar)
            ap_prob = torch.sigmoid(ap_l)[:, :, 0].cpu().numpy()
            ap_rate = expected_rate_mm_hr(ap_l, ap_i).cpu().numpy()
            b2_out = {}
            for variant, g in variants.items():
                logits, intensity = models["B2"](radar, g)
                b2_out[variant] = (torch.sigmoid(logits)[:, :, 0].cpu().numpy(), expected_rate_mm_hr(logits, intensity).cpu().numpy())
        for j, row in enumerate(event_rows.itertuples(index=False)):
            key = str(row.anchor_id)
            obs = np.where(payloads[j]["target_mask"] > 0, payloads[j]["target"], np.nan)
            latest = payloads[j]["radar"][-1]
            dry = np.isfinite(latest) & (latest < RAIN)
            cache[key] = {
                "obs": obs[:, dry],
                "obs_full": obs,
                "A_plus_prob": ap_prob[j][:, dry],
                "A_plus_rate": ap_rate[j],
            }
            for variant, (prob, rate) in b2_out.items():
                cache[key][f"B2_{variant}_prob"] = prob[j][:, dry]
                cache[key][f"B2_{variant}_rate"] = rate[j]
            rows.append({"anchor_id": key, "event_id": event_id, "event_class": row.event_class})
    return pd.DataFrame(rows), cache


def _aggregate(items: list[dict[str, float]]) -> dict[str, float]:
    return {k: float(np.nanmean([x[k] for x in items])) for k in items[0]}


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    index, cache = _predict_dev(output)
    positives = index[index.event_class == "positive_initiation"]
    negatives = index[index.event_class == "hard_negative_candidate"]
    sweep_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for model, field in (("A_plus", "A_plus_prob"), ("B2", "B2_normal_prob")):
        for threshold in GRID:
            per_event = []
            for event_id, group in positives.groupby("event_id"):
                obs = np.concatenate([cache[k]["obs"] for k in group.anchor_id], axis=1)
                prob = np.concatenate([cache[k][field] for k in group.anchor_id], axis=1)
                metrics = _lifecycle(obs, prob, threshold)
                per_event.append(metrics)
                event_rows.append({"model": model, "threshold": threshold, "event_id": event_id, **metrics})
            macro = _aggregate(per_event)
            neg_probs = np.concatenate([cache[k][field].ravel() for k in negatives.anchor_id])
            neg_obs = np.concatenate([cache[k]["obs"].ravel() for k in negatives.anchor_id])
            macro["hard_negative_fraction_above_threshold"] = float(np.mean(neg_probs >= threshold))
            macro["hard_negative_mean_probability"] = float(np.mean(neg_probs))
            macro["hard_negative_max_probability"] = float(np.max(neg_probs))
            macro["hard_negative_brier"] = brier_score((neg_obs >= RAIN).astype(float), neg_probs)
            accepted = macro["hard_negative_fraction_above_threshold"] <= 0.01 and macro["false_initiation_fraction"] <= 0.40
            sweep_rows.append({"model": model, "probability_threshold": threshold, "accepted": accepted, **macro})
    sweep = pd.DataFrame(sweep_rows)
    selected = {}
    for model in ("A_plus", "B2"):
        candidates = sweep[(sweep.model == model) & sweep.accepted].copy()
        if candidates.empty:
            candidates = sweep[sweep.model == model].copy()
        chosen = candidates.sort_values(["f1", "onset_mae_minutes", "brier_score"], ascending=[False, True, True]).iloc[0]
        selected[model] = float(chosen.probability_threshold)
    sweep.to_csv(output / "dev_common_probability_policy_sweep.csv", index=False)
    pd.DataFrame(event_rows).to_csv(output / "dev_initiation_metrics_by_event.csv", index=False)

    # Perturbation and cooling tests at B2's selected policy.
    perturb_rows = []
    for variant in (
        "normal",
        "global_shuffle",
        "similar_shuffle",
        "all_goes_zero",
        "cooling_zero",
        "raw_c13_zero",
    ):
        field = f"B2_{variant}_prob"
        for event_id, group in positives.groupby("event_id"):
            obs = np.concatenate([cache[k]["obs"] for k in group.anchor_id], axis=1)
            prob = np.concatenate([cache[k][field] for k in group.anchor_id], axis=1)
            perturb_rows.append({"variant": variant, "event_id": event_id, **_lifecycle(obs, prob, selected["B2"])})
    pd.DataFrame(perturb_rows).to_csv(output / "dev_b2_information_tests_by_event.csv", index=False)

    # Event-level deltas and bootstrap CIs. Initiation uses selected policies; broad fields use expected rates.
    comparisons = []
    event_values: dict[str, list[tuple[float, float]]] = {k: [] for k in ("initiation_f1", "false_initiation_fraction", "onset_mae", "csi_60", "csi_90", "brier")}
    for event_id, group in index.groupby("event_id"):
        keys = list(group.anchor_id)
        obs = np.concatenate([cache[k]["obs"] for k in keys], axis=1)
        if (group.event_class == "positive_initiation").all():
            a = _lifecycle(obs, np.concatenate([cache[k]["A_plus_prob"] for k in keys], axis=1), selected["A_plus"])
            b = _lifecycle(obs, np.concatenate([cache[k]["B2_normal_prob"] for k in keys], axis=1), selected["B2"])
            event_values["initiation_f1"].append((a["f1"], b["f1"]))
            event_values["false_initiation_fraction"].append((a["false_initiation_fraction"], b["false_initiation_fraction"]))
            event_values["onset_mae"].append((a["onset_mae_minutes"], b["onset_mae_minutes"]))
        for lead, name in zip(LEADS, ("csi_60", "csi_90")):
            a_rate = np.concatenate([cache[k]["A_plus_rate"][lead-1].ravel() for k in keys])
            b_rate = np.concatenate([cache[k]["B2_normal_rate"][lead-1].ravel() for k in keys])
            truth = np.concatenate([cache[k]["obs_full"][lead-1].ravel() for k in keys])
            event_values[name].append((categorical_metrics(truth, a_rate, RAIN)["csi"], categorical_metrics(truth, b_rate, RAIN)["csi"]))
        a_prob = np.concatenate([cache[k]["A_plus_prob"].ravel() for k in keys])
        b_prob = np.concatenate([cache[k]["B2_normal_prob"].ravel() for k in keys])
        truth = np.concatenate([cache[k]["obs"].ravel() for k in keys]) >= RAIN
        event_values["brier"].append((brier_score(truth, a_prob), brier_score(truth, b_prob)))
    rng = np.random.default_rng(SEED)
    for metric, pairs in event_values.items():
        delta = np.array([b-a for a,b in pairs], dtype=float)
        boot = np.array([np.nanmean(rng.choice(delta, len(delta), replace=True)) for _ in range(10000)])
        comparisons.append({"metric": metric, "events": len(delta), "mean_delta_b2_minus_aplus": np.nanmean(delta), "median_delta": np.nanmedian(delta), "fraction_b2_wins": np.nanmean(delta > 0) if metric not in {"false_initiation_fraction", "onset_mae", "brier"} else np.nanmean(delta < 0), "ci95_low": np.nanquantile(boot, .025), "ci95_high": np.nanquantile(boot, .975)})
    pd.DataFrame(comparisons).to_csv(output / "dev_event_bootstrap_deltas.csv", index=False)

    # Reliability bins; identity calibration is deliberately frozen because DEV has only five events.
    reliability = []
    for model, field in (("A_plus", "A_plus_prob"), ("B2", "B2_normal_prob")):
        p = np.concatenate([cache[k][field].ravel() for k in index.anchor_id])
        y = np.concatenate([(cache[k]["obs"].ravel() >= RAIN) for k in index.anchor_id])
        for lo in np.arange(0, 1, .1):
            mask = (p >= lo) & (p < lo + .1 if lo < .9 else p <= 1)
            reliability.append({"model": model, "bin_lower": lo, "count": int(mask.sum()), "mean_probability": float(np.mean(p[mask])) if mask.any() else np.nan, "observed_frequency": float(np.mean(y[mask])) if mask.any() else np.nan})
    pd.DataFrame(reliability).to_csv(output / "dev_reliability.csv", index=False)

    # Predeclared model decision: B2 must improve >=2/3 primary initiation criteria, >=2/3 broad criteria,
    # show normal > both perturbations in event-macro F1, and satisfy the same false-alarm constraints.
    chosen_rows = {m: sweep[(sweep.model == m) & (sweep.probability_threshold == t)].iloc[0] for m,t in selected.items()}
    a, b = chosen_rows["A_plus"], chosen_rows["B2"]
    primary = [b.f1 > a.f1, b.false_initiation_fraction < a.false_initiation_fraction, b.onset_mae_minutes < a.onset_mae_minutes]
    cmp = {r["metric"]: r for r in comparisons}
    broad = [cmp["csi_60"]["mean_delta_b2_minus_aplus"] > 0, cmp["csi_90"]["mean_delta_b2_minus_aplus"] > 0, cmp["brier"]["mean_delta_b2_minus_aplus"] < 0]
    pert = pd.DataFrame(perturb_rows).groupby("variant").f1.mean()
    dependence = bool(
        pert["normal"] > pert["global_shuffle"]
        and pert["normal"] > pert["similar_shuffle"]
        and pert["normal"] > pert["all_goes_zero"]
    )
    b2_selected = sum(primary) >= 2 and sum(broad) >= 2 and dependence and bool(b.accepted)
    decision = "B2_SELECTED" if b2_selected else "A_PLUS_NULL_GOES_RESULT"
    summary = {"decision": decision, "selected_model": "B2" if b2_selected else "A_plus", "selected_thresholds": selected, "calibration_method": "identity_no_posthoc_calibration", "primary_checks_b2_wins": [bool(x) for x in primary], "broad_checks_b2_wins": [bool(x) for x in broad], "satellite_dependence_passed": dependence, "selection_rule": {"constraints": {"hard_negative_fraction_above_threshold_max": .01, "false_initiation_fraction_max": .40}, "policy_order": ["maximize event-macro initiation F1", "minimize onset MAE", "minimize Brier"], "model_gate": "B2 wins at least 2/3 primary, 2/3 broad, normal beats shuffled and zeroed, and constraints pass"}, "dev_events": sorted(index.event_id.unique()), "final_holdout_evaluated": False}
    (output / "dev_final_decision.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_4b/final_dev_decision"))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
