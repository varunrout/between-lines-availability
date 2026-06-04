"""
Project-wide constants and configuration.

Competition IDs can be verified at runtime with:
    from statsbombpy import sb
    print(sb.competitions()[['competition_id','season_id','competition_name','season_name']])
"""

# ---------------------------------------------------------------------------
# Target competitions (all have StatsBomb 360 data)
# ---------------------------------------------------------------------------
COMPETITIONS = [
    {"competition_id": 55, "season_id": 43,  "name": "UEFA Euro 2020"},
    {"competition_id": 43, "season_id": 106, "name": "FIFA World Cup 2022"},
    {"competition_id": 55, "season_id": 282, "name": "UEFA Euro 2024"},
]

# ---------------------------------------------------------------------------
# Pitch dimensions (StatsBomb coordinate system)
# ---------------------------------------------------------------------------
PITCH_LENGTH = 120  # x-axis: 0 → 120
PITCH_WIDTH = 80    # y-axis: 0 → 80

# ---------------------------------------------------------------------------
# Build-up zone filter
# All events with ball_x <= BUILDUP_MAX_X are considered build-up phase.
# Attacking direction is standardised left-to-right (x increases toward goal).
# ---------------------------------------------------------------------------
BUILDUP_MAX_X = 80  # own/middle third; excludes final third (x > 80)

# Pitch zone x-boundaries
DEFENSIVE_THIRD_MAX_X = 40
MIDDLE_THIRD_MAX_X = 80
# Final third: x > 80

# Y-bands for centrality classification
CENTRAL_Y_MIN = 30
CENTRAL_Y_MAX = 50
HALF_SPACE_LOW_Y_MIN = 20
HALF_SPACE_LOW_Y_MAX = 30
HALF_SPACE_HIGH_Y_MIN = 50
HALF_SPACE_HIGH_Y_MAX = 60

# ---------------------------------------------------------------------------
# Event types to analyse
# ---------------------------------------------------------------------------
BUILDUP_EVENT_TYPES = ["Pass", "Carry", "Ball Receipt*"]

# Play patterns indicating set-pieces to exclude
SET_PIECE_PLAY_PATTERNS = [
    "From Corner",
    "From Free Kick",
    "From Throw In",
    "From Goal Kick",
    "From Kick Off",
]

# ---------------------------------------------------------------------------
# Opponent line detection
# ---------------------------------------------------------------------------
N_DEFENDERS_DEFENSIVE_LINE = 4   # deepest N outfield opponents → defensive line
N_DEFENDERS_MIDFIELD_LINE = 5    # next N opponents ahead of them → midfield line

# ---------------------------------------------------------------------------
# Findability thresholds
# ---------------------------------------------------------------------------
MAX_PASS_DISTANCE_M = 30.0   # max acceptable ball-to-receiver distance (StatsBomb metres ~1:1)
MIN_RECEIVER_SPACE_M = 2.0   # nearest defender must be at least this far from receiver
LANE_BLOCK_DISTANCE_M = 1.5  # defender within this distance of pass lane = blocked

# Findability score thresholds
FINDABLE_SCORE_THRESHOLD = 2  # score >= 2 → findable_option_available = 1

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
