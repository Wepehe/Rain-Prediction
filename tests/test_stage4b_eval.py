from pathlib import Path

import pytest

from ontario_nowcast.training.stage4b_eval import evaluate_stage4b


def test_stage4b_eval_refuses_final_initiation_holdout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="final initiation holdout"):
        evaluate_stage4b(
            Path("configs/experiments/stage_4b_b1_raw_c13.yaml"),
            checkpoint=tmp_path / "missing.pt",
            training_metadata=tmp_path / "missing.json",
            output_dir=tmp_path,
            splits=["stage4_initiation_holdout"],
        )
