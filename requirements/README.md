# Environments

`model.txt` records the package versions used by the H100 baseline environment. Install the PyTorch build compatible with the cluster CUDA driver and hardware.

`state-training.txt` is the training-only overlay for the pinned
`arc-state==0.11.3` source checkout. Install it while constraining against
`model.txt`, then install the checkout with `--no-deps`. It intentionally omits
`cell-eval`: the current upstream evaluator requires a newer AnnData release
than the registered model runtime. Keep evaluation in the separate scoring
environment instead of silently changing model dependencies.

`scoring.txt` freezes that separate Python 3.11 CPU evaluator environment.
After installing it, install the pinned `external/cell-eval2` checkout with
`pip install --no-deps -e external/cell-eval2`. The scoring launcher prefers
`.venv-score/bin/python` and also accepts an explicit `VCC_SCORE_PYTHON`.

`vcc-cli.txt` preserves the historical `vcc-cli==0.1.0` baseline. For the v10
packaging/upload scripts, use `vcc-cli-v10.txt` in a separate Python 3.11
environment: those scripts require `vcc-cli==0.2.0`. Do not install the official
client into the frozen training environment.

`public-flow-v10.txt` records the direct dependencies used for the public GO flow
and GenePT feature tests, including the pinned offline W&B SDK. Use a fresh
Python 3.11 environment rather than modifying an authenticated historical
runtime. This is not a complete transitive lock or a registered v11 training
environment; the GenePT model/runner remains a draft. Revalidate the PyTorch
CUDA build against the chosen compute node before any new training registration.

`vcc-download.txt` is the separate checksum-verification dependency for the
bounded H1 downloader. `tracking.txt` records the tracking-only SDK pin for
other launchers. Neither manifest grants permission for data transfers or
online W&B synchronization.

Tests are environment-specific, not a single combined dependency set. Use the
model/public-flow environment for model and feature tests, `.venv-score` for
`test_validate_public_cell_eval_run.py` and `test_state_seed_ensemble_scoring.py`,
the download environment for `test_acquire_vcc_h1.py`, and the official-client
environment for `test_submit_public_auxiliary_vcc_v10.py`. The latter uses
`unittest` and mocked API calls; no live submission is needed to test it.

`presentation.txt` contains only the packages used to build, inspect, and render the English strategy deck.
