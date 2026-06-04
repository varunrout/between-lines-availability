# Between-Lines Availability Model using StatsBomb 360 Data

This project analyzes build-up possessions to estimate **when the ball-carrier has a viable passing option between the opposition midfield and defensive lines**.

Rather than focusing only on completed line-breaking passes, the goal is to measure **availability**, **accessibility**, and eventually **decision quality**: was there a findable teammate between the lines at a given on-ball moment?

## Project question

During build-up possession, how often does the ball-carrier have a viable passing option between the opposition midfield and defensive lines?

## Why this project matters

This is a football analytics project focused on the options that exist *before* a pass is played.

It aims to answer questions such as:

- Which teams consistently create findable between-lines options?
- Which players are best at making themselves available in dangerous central pockets?
- Which ball-carriers recognize and use those options?
- How often do teams leave valuable central progression opportunities unused?

## Data

The project is built around:

- StatsBomb event data
- StatsBomb 360 freeze-frame data
- Match metadata and lineups where needed

## Unit of analysis

The core row in the dataset is:

> One on-ball event during a team possession in build-up phase.

Example fields:

| possession_id | event_id | team  | player_on_ball | event_type | x  | y  | between_lines_option_available |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 45 | abc123 | Spain | Busquets | Pass | 42 | 39 | 1 |
| 45 | def456 | Spain | Pedri | Carry | 51 | 44 | 0 |

Candidate event types include:

- Pass
- Carry
- Ball Receipt
- Pressure
- Miscontrol
- Dribble

## Build-up definition

A starting definition for build-up play:

> A possession phase where the team has controlled possession in its own half or middle third, before entering the final third.

Suggested filters:

- controlled possessions
- ball location in defensive or middle zones
- open play only
- exclude restarts initially
- standardize attacking direction left to right

Possible sub-phases:

- **First phase**: own third, goalkeeper/center-backs involved
- **Middle build-up**: progression through the middle zones
- **Advanced build-up**: just before final-third entry

## Defining “between the lines”

A practical working definition:

> A teammate is between the opposition midfield line and defensive line, in a central or half-space zone, and ahead of the ball.

For each 360 frame:

1. Identify defending players.
2. Standardize attacking direction.
3. Estimate the opposition defensive line.
4. Estimate the opposition midfield line.
5. Check whether any attacking teammate is positioned between those two lines.
6. Evaluate whether that teammate is actually findable.

Suggested rules:

| Concept | Rule |
| --- | --- |
| Defensive line | Median x-position of deepest 4 or 5 outfield defenders |
| Midfield line | Median x-position of next 3 to 5 defenders ahead of them |
| Between-lines zone | Area between midfield-line x and defensive-line x |
| Receiver candidate | Attacking teammate inside that zone |
| Ahead of ball | Receiver x greater than ball x |
| Centrality | Receiver y inside central or half-space band |

## Defining “findable”

A between-lines player is not automatically a viable option.

The project therefore introduces a **findability** concept based on whether the receiver can realistically be accessed by the ball-carrier.

Important feature groups:

| Feature group | Example features |
| --- | --- |
| Passing lane | Whether defenders block the lane from ball to receiver |
| Distance | Ball-to-receiver distance |
| Angle | Forwardness and pass angle |
| Ball pressure | Opponents near the ball-carrier |
| Receiver pressure | Defenders within 2m, 3m, or 5m of the receiver |
| Space | Distance to nearest defender |
| Zone | Central, half-space, or wide |
| Team shape | Number of teammates ahead of the ball |
| Opposition shape | Compactness and line-gap depth |
| Progression value | How much the option advances play |

A simple MVP rule:

A receiver is **findable** if:

- they are between the lines
- they are ahead of the ball
- pass distance is below a defined threshold
- they are not tightly marked
- the passing lane is not blocked
- they are inside the visible area

## Target variable

### Version A: Availability model

Primary target for the first version:

> Was there at least one findable between-lines option at this event?

Binary label:

- **1** = at least one findable between-lines player exists
- **0** = no findable between-lines player exists

### Version B: Pass selection model

Possible extension:

> Did the ball-carrier actually attempt to find the between-lines player?

This opens the door to analyzing decision-making and missed central access opportunities.

## Project workflow

### 1. Data extraction

Expected datasets:

- `events_df`: event-level sequence data
- `frames_df`: player locations for 360 events
- `visible_area_df`: visible pitch area information
- `matches_df`: match metadata
- `lineups_df`: player and team information

### 2. Possession and build-up filtering

Filter the sample to:

- open-play possessions
- controlled possessions
- pre-final-third events
- possessions with 360 data available

Useful engineered fields:

- `possession_start_x`
- `event_sequence_number`
- `possession_phase`
- `ball_zone`
- `under_pressure`

### 3. Opponent line detection

For each freeze frame:

1. Separate teammates and opponents.
2. Remove or separately handle the goalkeeper.
3. Sort defenders by x-position.
4. Estimate defensive and midfield line heights.
5. Measure the space between them.

Potential features:

- `defensive_line_x`
- `midfield_line_x`
- `line_gap_depth`
- `defensive_compactness_y`
- `vertical_compactness`

### 4. Receiver candidate detection

For each attacking teammate in frame, test whether the player:

- is ahead of the ball
- sits between the lines
- occupies a central or half-space zone
- is visible in the frame
- is not too tightly marked

Example receiver-level table:

| event_id | receiver | between_lines | distance_to_ball | nearest_defender | lane_blocked |
| --- | --- | ---: | ---: | ---: | ---: |

### 5. Passing lane model

For each ball-carrier to receiver pair:

- draw a line segment between ball and receiver
- measure defender distance to that line
- flag blocked or clear passing corridors

Potential features:

- `lane_blocked`
- `min_defender_distance_to_lane`
- `defenders_in_lane_corridor`
- `pass_distance`
- `pass_angle`

### 6. Findability score

Start with a rule-based score before moving to machine learning.

Example scale:

- **0** = not available
- **1** = between lines but not findable
- **2** = findable but risky
- **3** = clearly findable

### 7. Modeling

Initial model:

> Predict the probability that a build-up event contains a findable between-lines option.

Candidate models:

- Logistic Regression
- Random Forest
- XGBoost

Potential feature groups:

- ball location
- possession context
- ball pressure
- team shape
- opponent shape
- passing geometry
- receiver space

### 8. Outputs

#### Team-level metrics

- **Between-Lines Availability Rate**
- **Central Access Rate**
- **Half-Space Access Rate**
- **Missed Access Rate**
- **Line Gap Exploitation**

#### Player-level metrics

- **Receiver Availability**
- **Used Availability**
- **Ball-Carrier Recognition**
- **Progressive Option Creation**

## Visualizations

Potential visual outputs include:

- freeze-frame pitch plots
- between-lines zone overlays
- team heatmaps
- player availability maps
- possession-sequence animations
- team comparison bar charts
- availability vs usage scatter plots

One especially useful portfolio visual:

> A build-up event where the pass was not played, but the model identifies a high-probability between-lines option.

## MVP

The first version of the project should answer:

> How often does a team have a findable player between the lines during build-up?

### MVP scope

- StatsBomb events + 360 freeze frames
- rule-based between-lines detection
- passing lane obstruction logic
- team and player rankings for availability
- example freeze-frame visuals for available and unavailable options

## Portfolio framing

This project uses StatsBomb event and 360 freeze-frame data to identify moments in build-up play where a ball-carrier has a viable passing option between the opposition midfield and defensive lines.

Rather than only analyzing completed line-breaking passes, the model evaluates the **availability of between-lines receivers**, **passing lane accessibility**, **defensive compactness**, **pressure**, and **receiver space**.

The output can help measure:

- which teams create central progression options
- which players regularly make themselves available in dangerous pockets
- which ball-carriers recognize or ignore those options
- where valuable access opportunities are being missed
