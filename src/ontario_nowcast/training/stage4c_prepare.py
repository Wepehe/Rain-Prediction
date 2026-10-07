"""Validate Stage 4C tensors, compute train-only normalization, and plot overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIABLES = ("TMP_2m", "DPT_2m", "CAPE_surface", "CIN_surface", "PWAT_column")
TRANSFORMS = ("identity", "identity", "log1p_nonnegative", "signed_log1p_absolute", "identity")


def _transform(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    result[:, 2] = np.log1p(np.maximum(result[:, 2], 0))
    result[:, 3] = np.sign(result[:, 3]) * np.log1p(np.abs(result[:, 3]))
    return result


def run(output_dir: Path) -> dict[str, object]:
    manifest_path = Path("artifacts/stage_4c/tensors/stage4c_thermodynamic_tensor_manifest.csv")
    manifest = pd.read_csv(manifest_path)
    train = manifest[manifest.split.eq("train")]
    count = np.zeros(5, dtype=np.int64)
    total = np.zeros(5, dtype=np.float64)
    total_sq = np.zeros(5, dtype=np.float64)
    raw_min = np.full(5, np.inf)
    raw_max = np.full(5, -np.inf)
    for row in train.itertuples(index=False):
        payload = np.load(row.thermodynamic_tensor_path)
        raw = payload["thermodynamics"]
        mask = payload["thermodynamics_valid_mask"] > 0
        transformed = _transform(raw)
        for channel in range(5):
            valid = mask[:, channel] & np.isfinite(transformed[:, channel])
            values = transformed[:, channel][valid]
            count[channel] += values.size
            total[channel] += values.sum()
            total_sq[channel] += np.square(values).sum()
            raw_values = raw[:, channel][mask[:, channel] > 0]
            raw_min[channel] = min(raw_min[channel], float(raw_values.min()))
            raw_max[channel] = max(raw_max[channel], float(raw_values.max()))
    mean = total/count
    std = np.sqrt(np.maximum(total_sq/count-np.square(mean), 0))
    normalization = {
        "source_split": "train",
        "train_tensors": len(train),
        "variables": list(VARIABLES),
        "transforms": list(TRANSFORMS),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": count.tolist(),
        "raw_min": raw_min.tolist(),
        "raw_max": raw_max.tolist(),
        "clipping": None,
        "tensor_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    normalization_path = output_dir/"thermodynamic_normalization_train_only.json"
    normalization_path.write_text(json.dumps(normalization,indent=2),encoding="utf-8")
    (output_dir/"thermodynamic_normalization_train_only.sha256").write_text(hashlib.sha256(normalization_path.read_bytes()).hexdigest()+"\n",encoding="utf-8")

    requested = {
        "convective_initiation": "stage4_dev_initiation_2026_06_18",
        "organized_convection": "july_organized_convection_2025_07_13",
        "stratiform_precipitation": "september_stratiform_2024_09_24",
        "dissipation": "stage4_dev_dissipation_2026_05_20",
        "favourable_dry": "stage4_dev_dry_favourable_2026_07_10",
    }
    visual_rows = []
    visual_dir = output_dir/"visual_overlays"
    visual_dir.mkdir(parents=True,exist_ok=True)
    for label,event_id in requested.items():
        row = manifest[manifest.event_id.eq(event_id)].iloc[0]
        radar = np.load(row.radar_tensor_path)["target_rate_mm_hr"][0]
        thermo = np.load(row.thermodynamic_tensor_path)["thermodynamics"][0]
        fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
        panels=[(np.log1p(radar),"Radar log1p rate","turbo")]+[(thermo[i],f"{VARIABLES[i]} current","viridis" if i>1 else "coolwarm") for i in range(5)]
        for ax,(field,title,cmap) in zip(axes.ravel(),panels,strict=True):
            image=ax.imshow(field,origin="lower",cmap=cmap)
            ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(image,ax=ax,shrink=.75)
        path=visual_dir/f"{label}.png"
        fig.suptitle(f"{label}: {event_id}"); fig.savefig(path,dpi=130); plt.close(fig)
        visual_rows.append({"category":label,"event_id":event_id,"path":path.as_posix(),"finite":bool(np.isfinite(thermo).all()),"nonconstant_channels":int(sum(np.nanstd(thermo[i])>0 for i in range(5)))})
    pd.DataFrame(visual_rows).to_csv(output_dir/"visual_overlay_audit.csv",index=False)
    summary={"tensor_rows":len(manifest),"train_rows_used_for_normalization":len(train),"normalization_source_split":"train","visual_overlays":len(visual_rows),"all_visual_fields_finite":bool(all(x["finite"] for x in visual_rows)),"all_visual_channels_nonconstant":bool(all(x["nonconstant_channels"]==5 for x in visual_rows)),"training_started":False}
    (output_dir/"stage4c_spatial_normalization_gate.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=Path("artifacts/stage_4c/gate")); args=parser.parse_args(); print(json.dumps(run(args.output_dir),indent=2))


if __name__ == "__main__":
    main()
