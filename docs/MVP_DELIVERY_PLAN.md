# MVP Delivery Plan

This project should be delivered as a defensible football analytics MVP before adding heavier modelling.

## Project objective

Estimate, for each build-up on-ball event with StatsBomb 360 context, whether the ball-carrier has at least one findable teammate positioned between the opposition midfield and defensive lines.

The first version should not claim tactical truth, causal decision quality, or model-discovered opportunity value. It should claim a transparent rule-based availability signal that can be inspected, visualised and improved.

## Delivery phases

### Phase 1: Data contract

Inputs:

- StatsBomb events
- StatsBomb 360 freeze frames
- visible-area polygons
- match metadata
- optional lineup metadata

Required event-level outputs:

- `buildup_events.csv`
- `line_features.csv`
- `receiver_candidates.csv`
- `event_scores.csv`
- `merged_analysis.csv`
- `team_metrics.csv`
- `player_metrics.csv`

### Phase 2: Build-up sample

Keep the first MVP deliberately narrow:

- open-play possessions only
- Pass, Carry and Ball Receipt events
- standardised left-to-right attacking direction
- ball in defensive or middle third
- possessions/events with 360 context for receiver analysis

### Phase 3: Between-lines geometry

For each eligible event:

1. Identify opponents in the freeze frame.
2. Remove goalkeeper from line detection.
3. Estimate defensive line from deepest outfield defenders.
4. Estimate midfield line from the next opponent group.
5. Detect attacking teammates ahead of the ball and between those two lines.
6. Restrict valid receiver zones to central and half-space pockets.

### Phase 4: Findability signal

A candidate receiver receives a score from 0 to 3:

- +1 for reachable pass distance
- +1 for receiver separation from nearest defender
- +1 for clear passing lane

A candidate is only eligible for scoring if they are between the lines, ahead of the ball, visible and in a central or half-space zone.

The event-level target is:

```text
findable_option_available = 1 if any candidate score >= FINDABLE_SCORE_THRESHOLD
```

### Phase 5: Outputs

Team outputs:

- between-lines availability rate
- central access rate
- half-space access rate
- average line gap
- missed access rate, once pass-selection labels are added

Player outputs:

- receiver availability volume
- receiver availability rate
- ball-carrier recognition rate, once pass-selection labels are added

### Phase 6: Portfolio visuals

Prioritise visuals that explain the idea quickly:

- one event where the option is clearly available
- one event where no findable option exists
- team availability ranking
- player availability scatter
- pitch heatmap for a selected team

## Current PR focus

This PR stabilises the MVP foundation by:

- making empty or partial 360 samples safe
- enforcing central and half-space zones in the findability rule
- adding unit tests around geometry, findability and output metrics
- adding CI so every PR can run linting and tests

## Next PRs

1. Validate StatsBomb 360 loading on one competition with `--max-matches 1`.
2. Add a small fixture dataset so the end-to-end pipeline can be tested without remote data.
3. Add pass-selection labels for between-lines usage and missed opportunities.
4. Add report generation that turns outputs into a portfolio-ready markdown report.
5. Add a model only after the rule-based labels have been sanity checked visually.
