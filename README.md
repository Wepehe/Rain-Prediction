# Ontario precipitation nowcasting

Research code for probabilistic 0–120 minute precipitation nowcasting over Ontario. The central
hypothesis is that satellite and atmospheric-state information can add skill when rain has not yet
become visible to radar; the project therefore treats radar advection as a baseline, not the final
model.

The repository is currently at **Stage 0 / Milestone 1.5**. It provides a small, reproducible pilot
design, source manifests, event-mining logic, persistence, optical-flow, and optional PySTEPS
baselines, verification metrics, benchmark metadata, and data-access utilities. Large raw datasets
are intentionally excluded from Git.

## Quick start

Python 3.12 is recommended because meteorological GRIB dependencies do not yet consistently ship
Python 3.14 wheels.

```powershell
uv sync --python 3.12 --extra data --extra baseline --extra viz --extra dev
uv run pytest
uv run nowcast-estimate-storage --config configs/data/milestone_1.yaml
uv run nowcast-run-demo --output-dir artifacts/demo
uv run nowcast-run-benchmark --config configs/data/milestone_1_5.yaml
```

The demo generates a deterministic synthetic storm solely to exercise alignment, event mining,
baselines, metrics, and plotting. It is not reported as scientific model performance. Real sample
retrieval is handled separately so network availability never makes the test suite flaky.

## Data policy

- Raw downloads are immutable.
- Every download is recorded in a JSONL manifest with source URL, retrieval time, byte count, and
  SHA-256 hash.
- Missing radar scans remain missing; preprocessing never silently interpolates them.
- Event-level and chronological splits are required before model training.
- `data/`, `artifacts/`, and `experiments/` payloads are ignored by Git.

See [docs/data_sources.md](docs/data_sources.md) and
[docs/milestone_1_report.md](docs/milestone_1_report.md) for the source audit and first milestone.
The benchmark construction report is in
[docs/milestone_1_5_benchmark.md](docs/milestone_1_5_benchmark.md).

PySTEPS is declared as an optional non-Windows baseline dependency because the current Windows
Python 3.12 environment needs local C++ build tools to compile it. For the actual PySTEPS comparison,
use Python 3.11 with conda-forge or install the required Microsoft C++ Build Tools, then rerun
`nowcast-run-benchmark --include-pysteps`.
