# Contributing

Use Python 3.13 and `uv sync --locked`. Run Ruff lint/format checks, pytest and the offline CLI
pipeline before submitting a feature branch. Add behavioral tests for leakage, metric correctness,
serialization, constraints and provider boundaries. Never tune a router on held-out test outcomes.
Keep all strategy comparisons on identical complete observations and report missing data explicitly.
Do not commit secrets, raw downloads, runtime data or joblib binaries.

Original code contributions use MIT. Third-party benchmarks retain their own source terms;
do not assume that the code license permits redistribution of downloaded datasets.
