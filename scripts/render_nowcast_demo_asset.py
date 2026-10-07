"""Render the illustrative non-FINAL DEV dashboard image used in documentation."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ontario_nowcast.operational.visualization import dashboard_figure, load_forecast_npz


def main() -> None:
    with np.load("tests/fixtures/operational_residual_v1/dev_input.npz", allow_pickle=False) as data:
        history = data["history_rate"].astype(np.float32)
    product = load_forecast_npz("artifacts/operational/smoke_test_output/forecast.npz")
    figure = dashboard_figure(history, product, 60, show_mask=True, selected_point=(64, 64))
    target = Path("docs/assets/operational_nowcast_demo_60min.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(target)


if __name__ == "__main__":
    main()
