"""
Phase 4 – Between-lines receiver candidate detection.

For each build-up event that has 360 data, identifies which attacking
teammates qualify as potential between-lines receivers.

Usage
-----
    from src.features.receiver import detect_receiver_candidates

    receivers_df = detect_receiver_candidates(frames_df, buildup_df, line_df)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "shapely not installed – visible_area filtering will be skipped."
    )

from src.config import (
    CENTRAL_Y_MAX,
    CENTRAL_Y_MIN,
    HALF_SPACE_HIGH_Y_MAX,
    HALF_SPACE_HIGH_Y_MIN,
    HALF_SPACE_LOW_Y_MAX,
    HALF_SPACE_LOW_Y_MIN,
    PITCH_LENGTH,
    PITCH_WIDTH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_receiver_candidates(
    frames_df: pd.DataFrame,
    buildup_df: pd.DataFrame,
    line_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a candidate receiver table – one row per (event, teammate) pair.

    Parameters
    ----------
    frames_df:
        Tidy freeze-frame rows (one per player per event).
    buildup_df:
        Standardised build-up events with columns ``id``, ``x``, ``y``,
        ``_flip_frame``, ``team``.
    line_df:
        Line-detection output from ``src.features.line_detection``.

    Returns
    -------
    pd.DataFrame
        One row per candidate receiver with columns:
        ``event_id``, ``receiver_player_id``, ``receiver_player_name``,
        ``receiver_x``, ``receiver_y``, ``ahead_of_ball``,
        ``between_lines``, ``centrality_zone``, ``in_visible_area``,
        ``nearest_defender_dist``, ``distance_to_ball``.
    """
    # Standardise frame coordinates
    flip_info = buildup_df[["id", "_flip_frame", "x", "y"]].rename(
        columns={"id": "event_id", "x": "ball_x", "y": "ball_y"}
    )
    frames = frames_df.merge(flip_info, on="event_id", how="inner")

    if frames.empty:
        return pd.DataFrame()

    frames = _standardise_frame_coords(frames)

    # Merge line info
    frames = frames.merge(line_df, on="event_id", how="left")

    results: list[dict] = []
    for event_id, group in frames.groupby("event_id"):
        ball_x = group["ball_x"].iloc[0]
        ball_y = group["ball_y"].iloc[0]
        defensive_line_x = group["defensive_line_x"].iloc[0]
        midfield_line_x = group["midfield_line_x"].iloc[0]

        if pd.isna(defensive_line_x) or pd.isna(midfield_line_x):
            continue

        # Visible area polygon (may be None)
        visible_area_raw = group["visible_area"].iloc[0]
        flip_frame = bool(group["_flip_frame"].iloc[0])
        visible_polygon = _build_visible_polygon(visible_area_raw, flip_frame=flip_frame)

        # Opponent positions for nearest-defender calculation
        opponents = group[
            (~group["teammate"]) &
            (group["frame_x"].notna()) &
            (group["frame_y"].notna())
        ][["frame_x", "frame_y"]].values

        # Iterate over attacking teammates
        teammates = group[
            group["teammate"] &
            group["frame_x"].notna() &
            group["frame_y"].notna()
        ]

        for _, tm in teammates.iterrows():
            rx, ry = tm["frame_x"], tm["frame_y"]

            ahead_of_ball = rx > ball_x
            between_lines = (
                midfield_line_x < rx < defensive_line_x
            ) if defensive_line_x > midfield_line_x else False

            centrality_zone = _classify_centrality(ry)
            in_visible_area = _check_visible_area(rx, ry, visible_polygon)
            nearest_def = _nearest_defender_distance(rx, ry, opponents)
            dist_to_ball = float(np.hypot(rx - ball_x, ry - ball_y))

            results.append(
                {
                    "event_id": event_id,
                    "receiver_player_id": tm["player_id"],
                    "receiver_player_name": tm["player_name"],
                    "receiver_x": rx,
                    "receiver_y": ry,
                    "ball_x": ball_x,
                    "ball_y": ball_y,
                    "ahead_of_ball": ahead_of_ball,
                    "between_lines": between_lines,
                    "centrality_zone": centrality_zone,
                    "in_visible_area": in_visible_area,
                    "nearest_defender_dist": nearest_def,
                    "distance_to_ball": dist_to_ball,
                }
            )

    logger.info("Identified %d receiver candidates across events", len(results))
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _standardise_frame_coords(frames: pd.DataFrame) -> pd.DataFrame:
    frames = frames.copy()
    flip = frames["_flip_frame"].fillna(False).astype(bool)
    frames["frame_x"] = np.where(flip, PITCH_LENGTH - frames["frame_x"], frames["frame_x"])
    frames["frame_y"] = np.where(flip, PITCH_WIDTH - frames["frame_y"], frames["frame_y"])
    return frames


def _classify_centrality(y: float) -> str:
    """Return the y-zone name for a receiver's y-coordinate."""
    if CENTRAL_Y_MIN <= y <= CENTRAL_Y_MAX:
        return "central"
    if HALF_SPACE_LOW_Y_MIN <= y < HALF_SPACE_LOW_Y_MAX:
        return "half_space_low"
    if HALF_SPACE_HIGH_Y_MIN < y <= HALF_SPACE_HIGH_Y_MAX:
        return "half_space_high"
    return "wide"


def _check_visible_area(
    x: float,
    y: float,
    polygon: Optional[object],
) -> bool:
    """Return True if (x, y) falls within the visible area polygon."""
    if polygon is None or not SHAPELY_AVAILABLE:
        return True  # assume visible if no polygon available
    try:
        return bool(Point(x, y).within(polygon))
    except Exception:
        return True


def _build_visible_polygon(raw, flip_frame: bool = False) -> Optional[object]:
    """Build a Shapely Polygon from the raw visible_area coordinate list."""
    if not SHAPELY_AVAILABLE or raw is None:
        return None
    try:
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            coords = [
                (
                    PITCH_LENGTH - x if flip_frame else x,
                    PITCH_WIDTH - y if flip_frame else y,
                )
                for x, y in raw
            ]
            return Polygon(coords)
    except Exception:
        pass
    return None


def _nearest_defender_distance(rx: float, ry: float, opponents: np.ndarray) -> float:
    """Euclidean distance from receiver to the nearest opponent."""
    if len(opponents) == 0:
        return np.inf
    diffs = opponents - np.array([rx, ry])
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    return float(dists.min())
