"""Frozen DEV-only Stage 5 H1 decision evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from ..evaluation.metrics import brier_score, categorical_metrics, fractions_skill_score_km
from ..models.baselines import pysteps_deterministic_extrapolation
from ..models.frozen_residual_fusion import FrozenBaselineResidualFusion
from ..models.radar_convlstm import expected_rate_mm_hr
from ..models.radar_goes import build_stage4b_model
from .stage4b_holdout import _row_metrics
from .stage4b_lifecycle import _goes_inputs, _radar_inputs
from .stage4b_train import _load_npz
from .stage4c_eval import _bucket, _finish
from .stage5_data import Dataset

P_THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)
RAIN_THRESHOLDS = (0.1, 1.0, 2.5, 5.0)
LEADS = (30, 60, 90, 120)
H1_CONDITIONS = (
    "H1_normal",
    "H1_hrrr_zeroed",
    "H1_hrrr_shuffled",
    "H1_goes_zeroed",
    "H1_goes_shuffled",
    "H1_all_nonradar_zeroed",
    "H1_all_nonradar_shuffled",
)


def _load_b2(device):
    config = yaml.safe_load(Path("configs/experiments/stage_4b_b2_c13_cooling.yaml").read_text())
    model = build_stage4b_model(config).to(device)
    model.load_state_dict(
        torch.load("artifacts/stage_4b/b2_c13_cooling/model_best.pt", map_location=device)
    )
    model.eval()
    norm = json.loads(Path("artifacts/stage_4b/b2_c13_cooling/training_metadata.json").read_text())[
        "normalization"
    ]
    return model, norm


def _h1_inputs(example, shuffled):
    h, g = example.hrrr[None], example.goes[None]
    hs, gs = shuffled.hrrr[None], shuffled.goes[None]
    return {
        "H1_normal": (h, g),
        "H1_hrrr_zeroed": (torch.zeros_like(h), g),
        "H1_hrrr_shuffled": (hs, g),
        "H1_goes_zeroed": (h, torch.zeros_like(g)),
        "H1_goes_shuffled": (h, gs),
        "H1_all_nonradar_zeroed": (torch.zeros_like(h), torch.zeros_like(g)),
        "H1_all_nonradar_shuffled": (hs, gs),
    }


def run() -> dict:
    output = Path("artifacts/stage_5/h1/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    dataset = Dataset({"dev", "stage4_dev_repair"})
    examples = [dataset[i] for i in range(len(dataset))]
    order = np.roll(np.arange(len(examples)), 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h1 = FrozenBaselineResidualFusion().to(device)
    h1.load_state_dict(torch.load("artifacts/stage_5/h1/model_best.pt", map_location=device))
    h1.eval()
    b2, b2_norm = _load_b2(device)
    predictions = defaultdict(list)
    rates = defaultdict(list)
    deltas, observed, dry_masks = [], [], []
    with torch.no_grad():
        for index, example in enumerate(examples):
            base = [
                example.base_logits[None].to(device),
                example.base_intensity[None].to(device),
                example.base_expected[None].to(device),
            ]
            for condition, (hrrr, goes) in _h1_inputs(example, examples[order[index]]).items():
                logits, intensity, raw = h1(*base, hrrr.to(device), goes.to(device))
                predictions[condition].append(torch.sigmoid(logits)[0].cpu().numpy())
                rates[condition].append(expected_rate_mm_hr(logits, intensity)[0].cpu().numpy())
                if condition == "H1_normal":
                    deltas.append((4 * torch.tanh(raw[0, :, :4])).cpu().numpy())
            predictions["A_plus"].append(torch.sigmoid(example.base_logits).numpy())
            rates["A_plus"].append(example.base_expected.numpy())
            source = _load_npz(Path(example.metadata["radar_goes_tensor_path"]))
            radar_np = _radar_inputs(source, b2_norm)
            goes_np = _goes_inputs(source, b2_norm, channel_indices=[0, 1, 2, 3, 4])
            b2_logits, b2_intensity = b2(
                torch.from_numpy(radar_np[None]).to(device),
                torch.from_numpy(goes_np[None]).to(device),
            )
            predictions["B2"].append(torch.sigmoid(b2_logits)[0].cpu().numpy())
            rates["B2"].append(expected_rate_mm_hr(b2_logits, b2_intensity)[0].cpu().numpy())
            target = source["target"]
            observed.append(np.where(source["target_mask"] > 0, target, np.nan))
            radar = source["radar"]
            dry_masks.append(np.isfinite(radar[-1]) & (radar[-1] < 0.1))
            py_rate = pysteps_deterministic_extrapolation(radar[-3:], 20)
            rates["PySTEPS"].append(py_rate)
            predictions["PySTEPS"].append(
                (py_rate[:, None] >= np.asarray(RAIN_THRESHOLDS)[None, :, None, None]).astype(
                    np.float32
                )
            )
            hrrr_raw = np.load(example.metadata["stage5_cache_path"])["hrrr_apcp_hourly"]
            hourly = np.stack([hrrr_raw[0]] * 10 + [hrrr_raw[1]] * 10)
            rates["raw_HRRR"].append(hourly)
            predictions["raw_HRRR"].append(
                (hourly[:, None] >= np.asarray(RAIN_THRESHOLDS)[None, :, None, None]).astype(
                    np.float32
                )
            )
            print(f"[stage5-eval] {index + 1}/{len(examples)}", flush=True)

    positive = [
        i for i, x in enumerate(examples) if x.metadata["event_class"] == "positive_initiation"
    ]
    negatives = [
        i for i, x in enumerate(examples) if x.metadata["event_class"] == "hard_negative_candidate"
    ]
    # Fixed low-degree-of-freedom hybrid family.
    for weight in (0.0, 0.25, 0.5):
        name = f"hybrid_hrrr_weight_{weight:.2f}"
        for index in range(len(examples)):
            a = predictions["A_plus"][index]
            h = predictions["raw_HRRR"][index]
            predictions[name].append((1 - weight) * a + weight * h)
            rates[name].append(
                (1 - weight) * rates["A_plus"][index] + weight * rates["raw_HRRR"][index]
            )

    lifecycle_rows = []
    model_names = list(H1_CONDITIONS) + [
        "A_plus",
        "B2",
        "raw_HRRR",
        "PySTEPS",
        "hybrid_hrrr_weight_0.00",
        "hybrid_hrrr_weight_0.25",
        "hybrid_hrrr_weight_0.50",
    ]
    for name in model_names:
        for threshold in P_THRESHOLDS:
            for index in positive:
                dry = dry_masks[index]
                probability = predictions[name][index][:, 0]
                lifecycle_rows.append(
                    {
                        "model": name,
                        "probability_threshold": threshold,
                        "anchor_id": examples[index].metadata["anchor_id"],
                        "event_id": examples[index].metadata["event_id"],
                        **_row_metrics(observed[index][:, dry], probability[:, dry], threshold),
                    }
                )
    lifecycle = pd.DataFrame(lifecycle_rows)
    lifecycle.to_csv(output / "dev_lifecycle_by_row.csv", index=False)
    event = lifecycle.groupby(["model", "probability_threshold", "event_id"], as_index=False).mean(
        numeric_only=True
    )
    event.to_csv(output / "dev_lifecycle_by_event.csv", index=False)
    macro = event.groupby(["model", "probability_threshold"], as_index=False).mean(
        numeric_only=True
    )
    macro.to_csv(output / "dev_lifecycle_event_macro.csv", index=False)

    hard_rows = []
    for name in model_names:
        for threshold in P_THRESHOLDS:
            for index in negatives:
                probability = predictions[name][index][:, 0]
                valid = np.isfinite(observed[index])
                hard_rows.append(
                    {
                        "model": name,
                        "probability_threshold": threshold,
                        "anchor_id": examples[index].metadata["anchor_id"],
                        "event_id": examples[index].metadata["event_id"],
                        "mean_probability": float(np.mean(probability[valid])),
                        "max_probability": float(np.max(probability[valid])),
                        "wet_area_fraction": float(np.mean(probability[valid] >= threshold)),
                        "false_initiation_fraction": float(np.any(probability[valid] >= threshold)),
                        "brier_score": brier_score(
                            (observed[index][valid] >= 0.1).astype(float), probability[valid]
                        ),
                    }
                )
    hard = pd.DataFrame(hard_rows)
    hard.to_csv(output / "dev_hard_negative_by_row.csv", index=False)
    hard_event = hard.groupby(["model", "probability_threshold", "event_id"], as_index=False).mean(
        numeric_only=True
    )
    hard_event.to_csv(output / "dev_hard_negative_by_event.csv", index=False)
    hard_macro = hard_event.groupby(["model", "probability_threshold"], as_index=False).mean(
        numeric_only=True
    )
    hard_macro.to_csv(output / "dev_hard_negative_event_macro.csv", index=False)

    buckets = defaultdict(_bucket)
    for name in ("H1_normal", "A_plus", "B2", "raw_HRRR", "PySTEPS"):
        for index in range(len(examples)):
            for lead in LEADS:
                li = lead // 6 - 1
                for ti, threshold in enumerate(RAIN_THRESHOLDS):
                    metric = categorical_metrics(
                        observed[index][li], rates[name][index][li], threshold
                    )
                    bucket = buckets[(name, lead, threshold)]
                    for key in ("hits", "misses", "false_alarms", "correct_negatives"):
                        bucket[key] += metric[key]
                    bucket["brier"].append(
                        brier_score(
                            (observed[index][li] >= threshold).astype(float),
                            predictions[name][index][li, ti],
                        )
                    )
                    for radius in (6.0, 18.0, 36.0):
                        bucket[f"fss_{int(radius)}"].append(
                            fractions_skill_score_km(
                                observed[index][li], rates[name][index][li], threshold, radius, 2.0
                            )
                        )
    pd.DataFrame(
        [
            {"model": name, "lead_minutes": lead, "threshold_mm_hr": threshold, **_finish(bucket)}
            for (name, lead, threshold), bucket in sorted(buckets.items())
        ]
    ).to_csv(output / "dev_broad_metrics.csv", index=False)

    delta_rows = []
    for index, delta in enumerate(deltas):
        category = examples[index].metadata["event_class"]
        for lead in LEADS:
            values = delta[lead // 6 - 1]
            delta_rows.append(
                {
                    "anchor_id": examples[index].metadata["anchor_id"],
                    "event_id": examples[index].metadata["event_id"],
                    "category": category,
                    "lead_minutes": lead,
                    "mean_abs_delta_logit": float(np.mean(np.abs(values))),
                    "positive_fraction": float(np.mean(values > 0)),
                    "meaningfully_changed_fraction": float(np.mean(np.abs(values) >= 0.1)),
                }
            )
    pd.DataFrame(delta_rows).to_csv(output / "residual_diagnostics.csv", index=False)
    history = pd.read_csv("artifacts/stage_5/h1/training_history.csv")
    ax = history.plot(x="epoch", y=[c for c in history if "delta_logit" in c], marker="o")
    ax.set_ylabel("mean |delta logit|")
    ax.figure.tight_layout()
    ax.figure.savefig(output / "residual_magnitude_by_epoch.png", dpi=160)
    plt.close(ax.figure)

    joined = macro.merge(hard_macro, on=["model", "probability_threshold"], suffixes=("", "_hard"))
    eligible = joined[
        (joined.wet_area_fraction <= 0.01) & (joined.false_initiation_fraction_hard <= 0.40)
    ]
    h1_options = eligible[eligible.model == "H1_normal"]
    if h1_options.empty:
        selected_h1 = joined[joined.model == "H1_normal"].sort_values("f1", ascending=False).iloc[0]
        constraints_pass = False
    else:
        selected_h1 = h1_options.sort_values("f1", ascending=False).iloc[0]
        constraints_pass = True
    threshold = float(selected_h1.probability_threshold)
    at_threshold = macro[np.isclose(macro.probability_threshold, threshold)].set_index("model")
    a_plus_frozen = macro[
        (macro.model == "A_plus") & np.isclose(macro.probability_threshold, 0.35)
    ].iloc[0]
    a_f1 = float(a_plus_frozen.f1)
    h1_f1 = float(at_threshold.loc["H1_normal", "f1"])
    hybrid_candidates = eligible[eligible.model.str.startswith("hybrid_")]
    hybrid = (
        hybrid_candidates.sort_values("f1", ascending=False).iloc[0]
        if not hybrid_candidates.empty
        else joined[joined.model.str.startswith("hybrid_")]
        .sort_values("f1", ascending=False)
        .iloc[0]
    )
    losses = {name: h1_f1 - float(at_threshold.loc[name, "f1"]) for name in H1_CONDITIONS[1:]}
    gates = {
        "A_gain_over_A_plus_at_least_0.01": h1_f1 - a_f1 >= 0.01,
        "B_all_zero_degradation_at_least_0.005": losses["H1_all_nonradar_zeroed"] >= 0.005,
        "C_all_shuffle_degradation_at_least_0.005": losses["H1_all_nonradar_shuffled"] >= 0.005,
        "D_individual_modality_degradation_at_least_0.005": max(
            losses["H1_hrrr_zeroed"],
            losses["H1_hrrr_shuffled"],
            losses["H1_goes_zeroed"],
            losses["H1_goes_shuffled"],
        )
        >= 0.005,
        "E_gain_over_simple_hybrid_at_least_0.005": h1_f1 - float(hybrid.f1) >= 0.005,
        "F_hard_negative_constraints": constraints_pass,
    }
    result = {
        "selected_h1_threshold": threshold,
        "h1_event_macro_f1": h1_f1,
        "a_plus_frozen_threshold": 0.35,
        "a_plus_event_macro_f1_frozen_threshold": a_f1,
        "h1_minus_a_plus_f1": h1_f1 - a_f1,
        "best_simple_hybrid": {
            "model": hybrid.model,
            "threshold": float(hybrid.probability_threshold),
            "f1": float(hybrid.f1),
        },
        "destruction_f1_loss_at_h1_threshold": losses,
        "promotion_gates": gates,
        "promoted": all(gates.values()),
        "a_plus_checkpoint_unchanged": True,
        "protected_2023_holdout_opened": False,
    }
    (output / "decision_summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
