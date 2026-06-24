# Methodology Guardrails

These guardrails keep the project defensible while it is still an MVP.

## Safe claims

The current MVP can claim:

- A transparent rule-based estimate of whether a findable between-lines option existed.
- Team and player descriptive metrics from StatsBomb 360 freeze-frame context.
- Passing-lane and receiver-pressure proxies based on observed frame geometry.
- Visual examples that help an analyst inspect the rule.

## Claims to avoid

The current MVP should not claim:

- The true tactical intention of a player.
- Whether the ball-carrier definitely saw the option.
- Causal value created by a receiver.
- That an unplayed pass was definitely the best decision.
- That an ML classifier discovers availability independently if it is trained on rule-derived labels.

## Core definitions

### Between-lines receiver

A teammate is treated as a between-lines receiver only when they are:

- ahead of the ball
- between the estimated midfield and defensive lines
- in a central or half-space zone
- inside the visible area when visible-area data is available

### Findable receiver

A receiver becomes findable when they satisfy the base between-lines conditions and enough accessibility rules:

- pass distance is acceptable
- nearest defender is not too close
- passing lane is not blocked

### Event-level availability

An event is available when at least one receiver candidate reaches the configured findability threshold.

## Known limitations

### Line detection is approximate

Median x-position of grouped defenders is transparent and testable, but it is still a proxy. It can struggle with:

- transition moments
- very deep blocks
- man-oriented defensive structures
- partial 360 visibility
- moments where midfield and defensive lines are not clearly separable

### Passing lane is simplified

The current lane logic uses defender distance to the ball-to-receiver segment. This does not yet model player orientation, pass speed, receiver movement, technical ability or interception reach.

### Availability is not usage

A findable option is not the same as a completed pass or correct decision. Usage and missed-access metrics require a separate pass-selection label.

## Validation checklist

Before treating results as portfolio-ready, inspect:

1. Random available and unavailable freeze-frame examples.
2. Top and bottom teams by availability rate.
3. Whether wide players are excluded from the between-lines target.
4. Whether deep defensive blocks create sensible line gaps.
5. Whether team direction is standardised consistently across events and frames.
6. Whether missing 360 frames are excluded or clearly marked.
7. Whether player rankings pass basic football intuition checks.

## Recommended project wording

Use:

> The project estimates whether a ball-carrier has a viable between-lines option during build-up, using rule-based geometry from StatsBomb 360 freeze frames.

Avoid:

> The model proves which players made the right or wrong decision.
