"""
Phase 3 – Opponent line detection from 360 freeze frames.

For each freeze-frame event, estimates the positions of the opposition
defensive and midfield lines using the median x-position of sorted
outfield opponents.

Usage
-----
    from src.features.line_detection import detect_opponent_lines

    line_df = detect_opponent_lines(frames_df, buildup_df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import (
    N_DEFENDERS_DEFENSIVE_LINE,
    N_DEFENDERS_MIDFIELD_LINE,
    PITCH_LENGTH,
    PITCH_WIDTH,
)

logger = logging.getLogger(__name__)

GOALKEEPER_POSITION = "Goalkeeper"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_opponent_lines(
    frames_df: pd.DataFrame,
    buildup_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute opponent defensive and midfield line heights for each event.

    Parameters
    ----------
    frames_df:
        Tidy freeze-frame DataFrame (one row per player per event) as
        returned by ``src.data.loader._load_frames``.  Must have columns:
        ``event_id``, ``match_id``, ``teammate``, ``frame_x``, ``frame_y``,
        ``position_name``.
    buildup_df:
        Build-up events DataFrame as returned by
        ``src.features.buildup.filter_buildup``.  Must have columns:
        ``id`` (= event_id), ``match_id``, ``period``, ``team``,
        ``_flip_frame``, ``x``, ``y``.

    Returns
    -------
    pd.DataFrame
        One row per event_id with columns:
        ``event_id``, ``defensive_line_x``, ``midfield_line_x``,
        ``line_gap_depth``, ``defensive_compactness_y``,
        ``vertical_compactness``, ``n_opponents_in_frame``.
    """
    # Join flip flag from buildup events onto frames
    flip_info = buildup_df[["id", "_flip_frame"]].rename(columns={"id": "event_id"})
    frames = frames_df.merge(flip_info, on="event_id", how="inner")

    if frames.empty:
        logger.warning("No frames matched build-up events – check event_id join.")
        return pd.DataFrame()

    # Standardise freeze-frame coordinates using same flip flag as events
    frames = _standardise_frame_coords(frames)

    results: list[dict] = []
    for event_id, group in frames.groupby("event_id"):
        row = _compute_lines_for_event(event_id, group)
        if row is not None:
            results.append(row)

    if not results:
        return pd.DataFrame()

    line_df = pd.DataFrame(results)
    logger.info("Opponent lines computed for %d events", len(line_df))
    return line_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _standardise_frame_coords(frames: pd.DataFrame) -> pd.DataFrame:
    """Flip freeze-frame (x, y) for events where the team attacked R→L."""
    frames = frames.copy()
    flip = frames["_flip_frame"].fillna(False).astype(bool)
    frames["frame_x"] = np.where(flip, PITCH_LENGTH - frames["frame_x"], frames["frame_x"])
    frames["frame_y"] = np.where(flip, PITCH_WIDTH - frames["frame_y"], frames["frame_y"])
    return frames


def _compute_lines_for_event(
    event_id: str,
    group: pd.DataFrame,
) -> dict | None:
    """Estimate defensive and midfield line heights for a single event frame.

    The attacking team attacks left → right (increasing x) after
    standardisation, so the defending team's goal is at x = 120.
    Deepest defenders (smallest x among opponents) form the defensive line.
    """
    # Opponent outfield players only
    opponents = group[
        (~group["teammate"]) &
        (group["position_name"] != GOALKEEPER_POSITION) &
        (group["frame_x"].notna()) &
        (group["frame_y"].notna())
    ].copy()

    if len(opponents) < N_DEFENDERS_DEFENSIVE_LINE:
        # Not enough opponent data in frame
        return None

    # Sort by x ascending: lowest x = deepest defenders (furthest from opp goal)
    opponents_sorted = opponents.sort_values("frame_x")

    # Defensive line: median x of deepest N defenders
    def_players = opponents_sorted.iloc[:N_DEFENDERS_DEFENSIVE_LINE]
    defensive_line_x = float(np.median(def_players["frame_x"]))

    # Midfield line: median x of next N players ahead of the defensive line
    remaining = opponents_sorted.iloc[N_DEFENDERS_DEFENSIVE_LINE:]
    mid_players = remaining.iloc[:N_DEFENDERS_MIDFIELD_LINE]
    if mid_players.empty:
        midfield_line_x = defensive_line_x  # degenerate case
    else:
        midfield_line_x = float(np.median(mid_players["frame_x"]))

    # Derived metrics
    line_gap_depth = max(0.0, defensive_line_x - midfield_line_x)
    defensive_compactness_y = float(
        def_players["frame_y"].max() - def_players["frame_y"].min()
    ) if len(def_players) > 1 else 0.0
    vertical_compactness = line_gap_depth  # alias kept for clarity

    return {
        "event_id": event_id,
        "defensive_line_x": defensive_line_x,
        "midfield_line_x": midfield_line_x,
        "line_gap_depth": line_gap_depth,
        "defensive_compactness_y": defensive_compactness_y,
        "vertical_compactness": vertical_compactness,
        "n_opponents_in_frame": len(opponents),
    }
