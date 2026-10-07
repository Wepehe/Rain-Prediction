import hashlib
import json
from pathlib import Path


ROOT = Path("artifacts/stage_8/gate_predeclaration")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage8_gate_predeclaration_is_frozen_and_untrained():
    path = ROOT / "stage8_gate_predeclaration.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    expected = (ROOT / "stage8_gate_predeclaration.sha256").read_text().split()[0]
    assert _sha(path) == expected
    assert spec["status"] == "PREDECLARED_AMENDED_NOT_TRAINED"
    assert spec["created_before_gate_training"] is True
    assert spec["gate_trained"] is False
    assert spec["development_predictions_generated"] is False
    assert spec["final_features_generated"] is False
    assert spec["final_predictions_generated"] is False
    assert spec["consumed_2023_used"] is False


def test_stage8_gate_has_exact_declared_degrees_of_freedom():
    spec = json.loads((ROOT / "stage8_gate_predeclaration.json").read_text(encoding="utf-8"))
    expected_features = [
        "lead_fraction",
        "current_radar_wet_fraction",
        "a_plus_wet_fraction",
        "a_plus_mean_entropy",
        "pysteps_wet_fraction",
        "hrrr_wet_fraction",
        "a_plus_hrrr_disagreement_fraction",
        "hrrr_radar_forecast_support_fraction",
    ]
    assert [item["name"] for item in spec["features_in_order"]] == expected_features
    assert all(item["causal"] for item in spec["features_in_order"])
    assert spec["architecture"]["coefficient_count"] == 8
    assert spec["architecture"]["intercept_count"] == 1
    assert spec["architecture"]["hidden_layers"] == 0
    assert spec["architecture"]["l2_lambda"] == 1.0
    assert spec["forecast_formula"]["alpha"] == 0.5
    assert spec["development_validation"]["outer_positive_system_folds"] == 7
    assert len(spec["gate_output"]["leads_minutes"]) == 20


def test_stage8_amendment_uses_universal_nested_system_holdouts():
    spec = json.loads((ROOT / "stage8_gate_predeclaration.json").read_text(encoding="utf-8"))
    membership = spec["development_row_system_membership"]
    assert membership["rows"] == 272
    assert membership["positive_rows"] == 260
    assert membership["hard_negative_rows"] == 12
    assert membership["additional_negative_only_system_groups"] == []
    assert membership["all_hard_negative_systems_are_existing_positive_groups"] is True
    validation = spec["development_validation"]
    assert validation["outer_positive_system_folds"] == 7
    assert validation["additional_negative_only_outer_folds"] == 0
    assert len(validation["outer_folds"]) == 7
    for fold in validation["outer_folds"]:
        assert fold["outer_held_out_system"] not in fold["outer_training_systems"]
        assert len(fold["outer_training_systems"]) == 6
        assert len(fold["inner_folds"]) == 6
    assert "strictly positive" in spec["promotion"]["conditions"]["system_consistency"]
    assert spec["post_promotion_final_fit"]["ordinary_oof_threshold_role"] == "final model configuration only"


def test_stage8_dependencies_and_final_manifest_are_immutable():
    spec = json.loads((ROOT / "stage8_gate_predeclaration.json").read_text(encoding="utf-8"))
    assert _sha(Path(spec["qualification_freeze"]["path"])) == spec["qualification_freeze"]["sha256"]
    assert _sha(Path(spec["immutable_inputs"]["a_plus"]["checkpoint"])) == spec["immutable_inputs"]["a_plus"]["checkpoint_sha256"]
    assert _sha(Path(spec["final_set_policy"]["manifest"])) == spec["final_set_policy"]["manifest_sha256"]
    assert spec["final_set_policy"]["sealed"] is True
    assert spec["final_set_policy"]["features_before_development_freeze"] is False
    assert spec["final_set_policy"]["predictions_before_development_promotion"] is False
