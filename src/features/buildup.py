"""
Phase 2 – Build-up filtering and direction standardisation.

Takes the raw events DataFrame and returns a filtered, enriched copy that
contains only build-up phase events with standardised coordinates (all
teams attacking left → right, x increases toward goal).

Usage
-----
    from src.features.buildup import filter_buildup, standardise_direction

    buildup_df = filter_buildup(events_df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import (
    BUILDUP_EVENT_TYPES,
    BUILDUP_MAX_X,
    DEFENSIVE_THIRD_MAX_X,
    MIDDLE_THIRD_MAX_X,
    PITCH_LENGTH,
    PITCH_WIDTH,
    SET_PIECE_PLAY_PATTERNS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_buildup(events_df: pd.DataFrame) -> pd.DataFrame:
    """Return build-up phase events only, with standardised direction and
    enriched contextual columns.

    Steps applied
    -------------
    1. Standardise attacking direction (all teams attack left → right).
    2. Keep only target event types (Pass, Carry, Ball Receipt*).
    3. Exclude set-piece restarts.
    4. Keep only events where standardised ball x ≤ BUILDUP_MAX_X.
    5. Attach possession-level context fields.
    6. Attach event sequence number within possession.
    7. Classify ball zone and possession phase.

    Parameters
    ----------
    events_df:
        Raw events DataFrame as returned by ``src.data.loader.load_competition``.
        Must have columns: ``id``, ``match_id``, ``period``, ``team``,
        ``type``, ``play_pattern``, ``possession``, ``location``,
        ``under_pressure``.

    Returns
    -------
    pd.DataFrame
        Filtered, enriched events with added columns:
        ``x``, ``y``, ``ball_zone``, ``possession_phase``,
        ``event_sequence_number``, ``possession_start_x``,
        ``under_pressure_flag``.
    """
    df = events_df.copy()

    # ----- 1. Expand location into x/y (may already be split in flat attrs) -----
    df = _expand_location(df)

    # ----- 2. Standardise attacking direction -----
    direction_map = _build_direction_map(df)
    df = _apply_direction_standardisation(df, direction_map)

    # ----- 3. Filter event types -----
    df = df[df["type"].isin(BUILDUP_EVENT_TYPES)].copy()

    # ----- 4. Exclude set-piece play patterns -----
    if "play_pattern" in df.columns:
        df = df[~df["play_pattern"].isin(SET_PIECE_PLAY_PATTERNS)].copy()

    # ----- 5. Keep build-up x range -----
    df = df[df["x"] <= BUILDUP_MAX_X].copy()

    # ----- 6. Drop rows with missing location -----
    df = df.dropna(subset=["x", "y"]).copy()

    # ----- 7. Event sequence number within possession -----
    df = df.sort_values(["match_id", "possession", "index"]).copy()
    df["event_sequence_number"] = df.groupby(["match_id", "possession"]).cumcount() + 1

    # ----- 8. Possession start x -----
    possession_start = (
        df.groupby(["match_id", "possession"])["x"]
        .first()
        .rename("possession_start_x")
        .reset_index()
    )
    df = df.merge(possession_start, on=["match_id", "possession"], how="left")

    # ----- 9. Ball zone -----
    df["ball_zone"] = df["x"].apply(_classify_ball_zone)

    # ----- 10. Possession phase -----
    df["possession_phase"] = df["x"].apply(_classify_possession_phase)

    # ----- 11. Under-pressure flag (bool → int) -----
    df["under_pressure_flag"] = (
        df.get("under_pressure", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(int)
    )

    logger.info(
        "Build-up filter: %d events retained from %d total (%.1f%%)",
        len(df),
        len(events_df),
        100.0 * len(df) / max(len(events_df), 1),
    )
    return df.reset_index(drop=True)


def standardise_coordinates(
    x: float,
    y: float,
    flip: bool,
) -> tuple[float, float]:
    """Flip coordinates if team attacks right-to-left.

    StatsBomb pitch: x ∈ [0, 120], y ∈ [0, 80].
    When attacking right-to-left: new_x = 120 - x, new_y = 80 - y.
    """
    if flip:
        return PITCH_LENGTH - x, PITCH_WIDTH - y
    return x, y


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expand_location(df: pd.DataFrame) -> pd.DataFrame:
    """Parse ``location`` column into ``x`` and ``y`` if not already present."""
    if "x" in df.columns and "y" in df.columns:
        return df

    if "location" in df.columns:
        locs = df["location"].apply(
            lambda v: (v[0], v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else (np.nan, np.nan)
        )
        df = df.copy()
        df["x"] = locs.apply(lambda t: t[0])
        df["y"] = locs.apply(lambda t: t[1])
    else:
        # statsbombpy flatten_attrs=True produces location_x / location_y
        df = df.copy()
        df["x"] = df.get("location_x", np.nan)
        df["y"] = df.get("location_y", np.nan)
    return df


def _build_direction_map(df: pd.DataFrame) -> dict[tuple[int, int, str], bool]:
    """Determine whether each (match_id, period, team) plays right-to-left.

    Strategy
    --------
    In period 1 of each match the team that takes the kick-off attacks
    left → right (flip=False).  Their opponents attack right → left (flip=True).
    In period 2 and higher (odd vs even), directions reverse.

    Falls back to heuristic (average possession x) if kick-off cannot be found.

    Returns
    -------
    dict mapping (match_id, period, team) → flip_bool
    """
    direction_map: dict[tuple[int, int, str], bool] = {}

    for match_id, match_df in df.groupby("match_id"):
        kickoff_p1 = match_df[
            (match_df["period"] == 1) & (match_df["type"] == "Kick Off")
        ]
        if not kickoff_p1.empty:
            kickoff_team = kickoff_p1.iloc[0]["team"]
            other_teams = [t for t in match_df["team"].unique() if t != kickoff_team]
        else:
            # Fallback: team with higher avg x in period 1 attacks right
            p1 = match_df[match_df["period"] == 1]
            p1_avg_x = _expand_location(p1).groupby("team")["x"].mean()
            if not p1_avg_x.empty:
                kickoff_team = p1_avg_x.idxmax()
                other_teams = [t for t in p1_avg_x.index if t != kickoff_team]
            else:
                continue

        teams = match_df["team"].unique().tolist()
        periods = match_df["period"].unique().tolist()

        for period in periods:
            # Directions swap each period
            for team in teams:
                is_kickoff_team = team == kickoff_team
                # Period 1: kickoff team → no flip (attacks L→R)
                # Period 2: kickoff team → flip (attacks R→L)
                # Period 3: back to no flip, etc.
                no_flip_in_odd = is_kickoff_team
                flip = not no_flip_in_odd if (period % 2 == 1) else no_flip_in_odd
                direction_map[(match_id, period, team)] = flip

    return direction_map


def _apply_direction_standardisation(
    df: pd.DataFrame,
    direction_map: dict[tuple[int, int, str], bool],
) -> pd.DataFrame:
    """Apply (or not) coordinate flipping based on the direction map."""
    df = _expand_location(df)
    df = df.copy()

    flip_flags = df.apply(
        lambda row: direction_map.get((row["match_id"], row["period"], row["team"]), False),
        axis=1,
    )
    df["_flip"] = flip_flags

    df["x"] = np.where(df["_flip"], PITCH_LENGTH - df["x"], df["x"])
    df["y"] = np.where(df["_flip"], PITCH_WIDTH - df["y"], df["y"])
    df["_flip_frame"] = df["_flip"]  # carry for later use in frame standardisation

    df = df.drop(columns=["_flip"])
    return df


def _classify_ball_zone(x: float) -> str:
    """Classify standardised x-position into pitch zone."""
    if x <= DEFENSIVE_THIRD_MAX_X:
        return "defensive_third"
    if x <= MIDDLE_THIRD_MAX_X:
        return "middle_third"
    return "final_third"


def _classify_possession_phase(x: float) -> str:
    """Classify build-up sub-phase from standardised x-position."""
    if x <= DEFENSIVE_THIRD_MAX_X:
        return "first_phase"
    if x <= MIDDLE_THIRD_MAX_X:
        return "middle_buildup"
    return "advanced_buildup"
