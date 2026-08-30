# Contributing

This is a private three-person competition repository. Changes should remain reproducible, reviewable, and safe for the limited VCC submission quota.

## Language

Use English for all code, comments, docstrings, documentation, branch names, commit messages, issues, pull requests, and review comments.

## Branches and commits

- Keep `main` releasable.
- Use short-lived branches named `feat/*`, `fix/*`, `exp/*`, or `docs/*`.
- Use English Conventional Commit messages such as `feat: add target-cluster cross-validation`.
- Do not force-push shared branches.
- Prefer squash merges after review.

## Pull requests

Every modeling pull request must include:

- the biological or engineering hypothesis;
- data sources and license constraints;
- split definition and leakage controls;
- exact configuration and random seed;
- commit and input checksums;
- Slurm job identifiers and hardware;
- component metrics, not only the overall score;
- failure modes and rollback instructions.

At least one non-author teammate should approve a modeling or submission-path change.

## Repository hygiene

Never commit:

- VCC or GitHub credentials;
- `.env` files or private keys;
- challenge controls or held-out data;
- public support matrices that cannot be redistributed;
- model checkpoints, prediction matrices, `.vcc` files, or scheduler logs;
- notebooks with saved secrets or personal operational output.

Before pushing, inspect the staged files and run the checks documented in `docs/REPRODUCIBILITY.md`.

## Experiments

An experiment is reproducible only when it records the code commit, configuration, seed, data hashes, software versions, hardware, wall time, output hashes, and all available evaluation components. Leaderboard-only optimization is not accepted as local validation.

## Submissions

Only the designated submitter may call `vcc submit`. Each candidate must pass internal schema checks, scientific QC, official `vcc prep --dry-run`, package validation, and checksum verification. Interrupted uploads must use `--resume`; they must not create a second entry.
