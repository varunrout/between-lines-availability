# CI/CD Workflows

This repository uses a small set of workflows that separate fast pull-request checks from heavier data and release jobs.

## CI

### `CI`

File: `.github/workflows/ci.yml`

Runs on pull requests and pushes to `main`.

Checks:

- installs the package in editable mode
- runs undefined-name linting with Ruff
- runs pytest
- validates the package against Python 3.10, 3.11 and 3.12

This is the default PR gate.

### `Notebook Check`

File: `.github/workflows/notebook-check.yml`

Runs when notebooks change.

Checks:

- notebook files are valid JSON
- notebooks contain a valid `cells` list

This avoids broken notebook files entering the repo without forcing heavy notebook execution on every PR.

### `Pipeline Smoke`

File: `.github/workflows/pipeline-smoke.yml`

Manual workflow. Use it after methodology or data-loading changes.

Default command equivalent:

```bash
python scripts/run_pipeline.py --competitions euro2024 --max-matches 1 --output-dir outputs/smoke --skip-model
```

It uploads the generated smoke outputs as a GitHub Actions artifact.

## CD

### `Build Artifacts`

File: `.github/workflows/build-artifacts.yml`

Runs on pushes to `main` and manually.

Builds:

- source distribution
- wheel distribution

Uploads the package distributions as GitHub Actions artifacts.

### `Release`

File: `.github/workflows/release.yml`

Runs when a version tag such as `v0.1.0` is pushed, or manually with an existing tag.

Publishes:

- GitHub release notes
- source distribution
- wheel distribution

## Recommended usage

Use the workflows in this order:

1. Every PR: `CI`
2. Notebook PRs: `Notebook Check`
3. Data/methodology PRs: run `Pipeline Smoke` manually before merging
4. After milestone merge to `main`: run `Build Artifacts`
5. For portfolio-ready milestones: tag the repo and run `Release`

## Future hardening

Once the MVP stabilises, add:

- fixture-based end-to-end tests that do not rely on remote StatsBomb calls
- stricter Ruff checks for import order and unused imports
- coverage thresholds
- generated markdown report artifacts
- optional scheduled weekly data smoke run
