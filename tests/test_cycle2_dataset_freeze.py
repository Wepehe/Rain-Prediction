import json
from pathlib import Path
import pandas as pd
import pytest

from ontario_nowcast.training.cycle2_data import extract_radar_window, load_nonfinal_rows

ROOT=Path("artifacts/cycle2/data")


@pytest.fixture(scope="module")
def frozen():
    required=[ROOT/"frozen_system_split.csv",ROOT/"train_dev_rows.csv",ROOT/"final_manifest.json"]
    if not all(p.exists() for p in required):pytest.skip("Cycle-2 freeze has not run")
    return pd.read_csv(required[0]),pd.read_csv(required[1]),json.loads(required[2].read_text())


def test_system_split_gate_and_no_leakage(frozen):
    systems,_,_=frozen
    assert len(systems)>=60
    assert systems.system_id.nunique()==len(systems)
    assert (systems.split=="new_dev").sum()==12
    assert (systems.split=="new_final").sum()==12
    dates=sorted(pd.to_datetime(systems.date))
    assert min((b-a).days for a,b in zip(dates,dates[1:]))>=3


def test_residual_relevant_qualification_rule(frozen):
    systems,_,_=frozen;audit=pd.read_csv(ROOT/"all_system_row_audit.csv")
    qualified=audit[audit.qualification_status=="qualified"]
    assert (qualified.active_candidates>=12).all()
    assert (audit.loc[audit.qualification_status=="rejected","active_candidates"]<12).all()
    assert (qualified.clean_candidates<3).any(), "clean initiation still appears to gate qualification"
    forbidden=[c for c in audit.columns if "pysteps" in c.lower() or "model" in c.lower() or "skill" in c.lower()]
    assert not forbidden


def test_training_support_constraints(frozen):
    systems,_,_=frozen
    for season,g in systems.groupby("season"):
        counts=g.split.value_counts()
        if season=="winter" and len(g)>=7:
            assert counts.get("new_train",0)>=3 and counts.get("new_dev",0)>=2 and counts.get("new_final",0)>=2
        elif len(g)>=6:
            assert counts.get("new_train",0)>=2 and counts.get("new_dev",0)>=1 and counts.get("new_final",0)>=1
    for _,g in systems.groupby("regime"):
        if len(g)>=2:assert (g.split=="new_train").any()


def test_legacy_never_enters_new_dev_or_final(frozen):
    systems,_,_=frozen;legacy=json.loads((ROOT/"legacy_training_only.json").read_text())
    dates={x["date"] for x in legacy["events"]}
    assert not set(systems.loc[systems.split.isin(["new_dev","new_final"]),"date"].astype(str))&dates


def test_nonfinal_manifests_exclude_final(frozen):
    _,rows,_=frozen
    assert "new_final" not in set(rows.split)
    negatives=pd.read_csv(ROOT/"train_dev_hard_negatives.csv")
    assert "new_final" not in set(negatives.split)
    assert set(rows.row_type)<={"clean_initiation","active_precip","hard_negative"}
    counts=rows.groupby(["system_id","row_type"]).size().unstack(fill_value=0)
    assert (counts.get("clean_initiation",0)<=12).all()
    assert (counts.get("active_precip",0)<=24).all()
    assert (counts.get("hard_negative",0)<=1).all()


def test_row_geometry_timing_and_loader_guards(frozen):
    _,rows,_=frozen
    assert ((rows.y1-rows.y0+1)==128).all() and ((rows.x1-rows.x0+1)==128).all()
    sample=rows.iloc[0];history,target,_=extract_radar_window(sample)
    assert history.shape==(10,128,128) and target.shape==(20,128,128)
    forged=sample.copy();forged["split"]="new_final"
    with pytest.raises(PermissionError):extract_radar_window(forged)
    assert "new_final" not in set(load_nonfinal_rows(role="train").split)
    assert "new_final" not in set(load_nonfinal_rows(role="dev").split)


def test_active_rows_are_issue_time_selected(frozen):
    active=pd.read_csv(ROOT/"active_precip_rows.csv")
    assert set(active.selection_basis)=={"current_radar_only"}
    assert (active.current_wet_fraction>0).all() and (active.current_max_rate>.1).all()


def test_hard_negative_frozen_dry_criteria(frozen):
    negatives=pd.read_csv(ROOT/"train_dev_hard_negatives.csv")
    assert (negatives.history_target_mean_wet_fraction<=.001+1e-12).all()
    assert (negatives.history_target_max_wet_fraction<=.005+1e-12).all()
    assert (negatives.history_target_heavy_fraction==0).all()
    assert (negatives.validity_fraction>=.9).all()
    assert (negatives.current_validity_fraction>=.9).all()
    assert negatives.current_max_rate.notna().all()


def test_final_manifest_is_sealed(frozen):
    _,_,manifest=frozen
    assert manifest["status"]=="SEALED_UNSCORED"
    assert manifest["model_predictions_generated"] is False
    assert manifest["future_targets_materialized_for_scoring"] is False
    assert manifest["supersedes"]["status"]=="SUPERSEDED_PRETRAINING"
    assert manifest["supersedes"]["final_manifest_sha256"]=="32d83c5c69e07cee1b5f072892500c05dbaf20679bb027b1d248b2555c64196e"
    assert not list(Path("artifacts/cycle2").glob("**/replacement_final*prediction*"))
