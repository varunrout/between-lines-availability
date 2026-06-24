# Completion PR Delivery Plan

The project should be completed through small, reviewable PRs. Each PR should leave the repo runnable and defensible.

## Current baseline

### PR 1: Stabilise MVP foundation

Status: open as PR #2.

Goal:

Make the existing implementation safe enough to build on.

Delivered in this branch:

- stable findability scoring
- robust team and player metric outputs
- CI across supported Python versions
- notebook validation
- manual pipeline smoke workflow
- build and release workflows
- methodology and delivery documentation

Merge criteria:

- CI passes
- PR is mergeable
- README smoke-run command is clear

## Completion sequence

### PR 2: Data contract and fixture end-to-end test

Goal:

Make the project testable without relying on remote StatsBomb calls.

Deliverables:

- small synthetic event and 360 frame fixture
- fixture builder under `tests/fixtures/`
- end-to-end test from build-up filtering to team/player metrics
- explicit schema contract for core outputs
- documented expected columns and row grains

Acceptance criteria:

- `pytest` validates the full pipeline on fixture data
- missing 360 events are handled deliberately
- event-level and receiver-level grains are documented

### PR 3: StatsBomb smoke validation and data ingestion hardening

Goal:

Validate that the pipeline runs against a small real StatsBomb sample.

Deliverables:

- run `Pipeline Smoke` for `euro2024` with `max_matches=1`
- fix loader issues caused by StatsBomb API/frame format differences
- add raw-data caching option where appropriate
- add clear logs for loaded events, frames and dropped rows
- add failure notes for competitions with unavailable 360 data

Acceptance criteria:

- manual smoke workflow produces artifacts
- `buildup_events.csv`, `merged_analysis.csv`, `team_metrics.csv` and `player_metrics.csv` are generated
- no silent empty-output failure

### PR 4: Methodology validation visuals

Goal:

Make the rule-based availability signal visually auditable.

Deliverables:

- example available event plot
- example unavailable event plot
- line-gap sanity plots
- receiver candidate overlays
- visual QA notes in markdown
- artifact upload from the smoke workflow

Acceptance criteria:

- at least 5 inspected examples are documented
- wide receivers are visibly excluded from the final between-lines target
- defensive and midfield line estimates look defensible in sampled events

### PR 5: Pass-selection and missed-access labels

Goal:

Move from availability only to availability versus usage.

Deliverables:

- label whether the actual pass targeted a between-lines candidate
- candidate matching logic between pass recipient and freeze-frame teammate
- `pass_to_between_lines` event field
- missed-access metrics for teams and ball-carriers
- tests for pass-recipient matching edge cases

Acceptance criteria:

- missed-access rate is populated when pass data supports it
- non-pass events do not produce misleading usage labels
- player recognition metrics are clearly marked as usage-based proxies

### PR 6: Portfolio report generation

Goal:

Turn outputs into a readable project artifact.

Deliverables:

- `scripts/generate_report.py`
- `outputs/reports/between_lines_report.md`
- team ranking table
- player ranking table
- selected freeze-frame examples
- limitations and next-step sections
- workflow artifact for generated reports

Acceptance criteria:

- report can be regenerated from outputs
- report does not make unsupported tactical claims
- visual and tabular outputs are portfolio-ready

### PR 7: Availability model and model validation

Goal:

Add ML only after the rule-based label has been validated.

Deliverables:

- cleaner train/validation split by match
- logistic baseline
- optional XGBoost model
- calibration and ROC/PR metrics
- feature importance report
- clear note that the model predicts rule-derived availability labels

Acceptance criteria:

- no event leakage across train/test split
- model metrics are produced only when enough labelled rows exist
- model purpose is framed as prediction of the availability proxy, not proof of tactical truth

### PR 8: Packaging and release milestone

Goal:

Create the first portfolio-ready release.

Deliverables:

- version bump to `0.1.0`
- release notes
- generated package artifact
- final README polish
- final methodology guardrail review

Acceptance criteria:

- all CI passes
- smoke workflow has a successful run
- release workflow creates GitHub release artifacts

## Suggested milestones

### Milestone 1: MVP defensible foundation

Includes PRs 1 to 3.

Outcome:

The repo is installable, testable and can run on at least one small real-data sample.

### Milestone 2: Analyst-ready output

Includes PRs 4 to 6.

Outcome:

The project can explain availability with plots, metrics and a generated report.

### Milestone 3: Model-ready portfolio project

Includes PRs 7 to 8.

Outcome:

The project has a documented predictive layer, release artifacts and a portfolio-ready final state.

## Branch naming convention

Use:

- `codex/fixture-e2e-contract`
- `codex/statsbomb-smoke-hardening`
- `codex/methodology-validation-visuals`
- `codex/pass-selection-labels`
- `codex/report-generation`
- `codex/availability-model-validation`
- `codex/release-v0-1-0`

## Definition of done for every PR

Every PR should include:

- focused code change
- tests or a clear reason tests are not applicable
- README or docs update if user-facing behaviour changes
- CI passing
- clear limitation notes if the method still relies on approximations
