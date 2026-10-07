"""Guarded access and TRAIN-only normalization for the frozen Cycle-2 corpus."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from ..sample_evaluation import _downsample_max

ROOT = Path("artifacts/cycle2/data")


def load_nonfinal_rows(*, role: str, root: Path = ROOT) -> pd.DataFrame:
    if role not in {"train", "dev"}:
        raise ValueError("role must be train or dev; final access is not authorized")
    rows = pd.read_csv(root / "train_dev_rows.csv")
    if (rows["split"] == "new_final").any():
        raise RuntimeError("sealed NEW FINAL row found in non-final manifest")
    wanted = "new_train" if role == "train" else "new_dev"
    return rows.loc[rows["split"] == wanted].reset_index(drop=True)


def extract_radar_window(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exactly 10 history and 20 future frames for a non-final row."""
    if row["split"] == "new_final":
        raise PermissionError("NEW FINAL tensors are sealed")
    with np.load(Path(row["source_path"])) as data:
        times = pd.to_datetime(data["times"], utc=True)
        issue = pd.Timestamp(row["issue_time_utc"])
        anchor = int(times.get_indexer([issue])[0])
        if anchor < 9 or anchor + 20 >= len(times):
            raise ValueError("row does not support 10 history and 20 future frames")
        y0,y1,x0,x1=(int(row[k]) for k in ("y0","y1","x0","x1"))
        values=_downsample_max(data["rate_mm_hr"],2)[:,y0:y1+1,x0:x1+1]
    history=values[anchor-9:anchor+1];future=values[anchor+1:anchor+21]
    if history.shape!=(10,128,128) or future.shape!=(20,128,128):
        raise ValueError(f"unexpected Cycle-2 tensor shapes: {history.shape}, {future.shape}")
    return history,future,np.isfinite(history)


def fit_train_rate_normalization(*, root: Path = ROOT) -> dict[str, float]:
    """Fit log-rate moments exclusively from NEW TRAIN rows."""
    rows=load_nonfinal_rows(role="train",root=root);count=0;total=0.0;total2=0.0
    for _,row in rows.iterrows():
        history,_,valid=extract_radar_window(row);x=np.log1p(np.clip(history[valid],0,None)).astype(np.float64)
        count+=x.size;total+=float(x.sum());total2+=float(np.square(x).sum())
    mean=total/count;variance=max(total2/count-mean*mean,1e-12)
    return {"rate_log1p_mean":mean,"rate_log1p_std":variance**0.5,"samples":count,"source_split":"new_train"}
