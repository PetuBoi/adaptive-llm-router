# Engineering rules

Use Python 3.13 and uv. Run Ruff and pytest before committing. Never use held-out responses,
quality, latency or token usage as inference features. Oracle belongs only in offline evaluation.
Group related prompts, fit transforms on training only, and compare strategies on identical
complete test observations. Keep estimates separate from observations. Label all demo results
SIMULATED. Never fabricate real metrics, costs, live responses or pricing. No arbitrary endpoints.
Keep secrets, raw data and model binaries out of Git.
