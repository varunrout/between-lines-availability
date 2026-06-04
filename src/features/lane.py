"""
Phase 5 – Passing lane obstruction.

For each ball-carrier → receiver candidate pair, determines whether the
passing lane is blocked by a defending player.

Usage
-----
    from src.features.lane import compute_lane_features

    lane_df = compute_lane_features(receivers_df, frames_df, buildup_df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import LANE_BLOCK_DISTANCE_M, PITCH_LENGTH, PITCH_WIDTH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_lane_features(
    receivers_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    buildup_df: pd.DataFrame,
) -> pd.DataFrame:
    """Augment receiver candidates with passing lane obstruction features.

    Parameters
    ----------
    receivers_df:
        Output of ``src.features.receiver.detect_receiver_candidates``.
        Must have columns: ``event_id``, ``ball_x``, ``ball_y``,
        ``receiver_x``, ``receiver_y``.
    frames_df:
        Tidy freeze-frame rows with ``event_id``, ``teammate``,
        ``frame_x``, ``frame_y``.
    buildup_df:
        Build-up events with ``id`` → ``event_id``, ``_flip_frame``.

    Returns
    -------
    pd.DataFrame
        ``receivers_df`` with added columns:
        ``lane_blocked``, ``min_lane_defender_dist``,
        ``defenders_in_corridor``, ``pass_distance``, ``pass_angle_deg``.
    """
    if receivers_df.empty:
        return receivers_df.copy()

    # Build a lookup: event_id → opponent (x, y) array (standardised)
    flip_info = buildup_df[["id", "_flip_frame"]].rename(columns={"id": "event_id"})
    frames = frames_df.merge(flip_info, on="event_id", how="inner")
    frames = _standardise_frame_coords(frames)

    opponents_by_event: dict[str, np.ndarray] = {}
    for event_id, group in frames.groupby("event_id"):
        opp = group[
            (~group["teammate"]) &
            group["frame_x"].notna() &
            group["frame_y"].notna()
        ][["frame_x", "frame_y"]].values
        opponents_by_event[event_id] = opp

    # Compute lane features row by row
    results = []
    for _, row in receivers_df.iterrows():
        eid = row["event_id"]
        bx, by = row["ball_x"], row["ball_y"]
        rx, ry = row["receiver_x"], row["receiver_y"]
        opponents = opponents_by_event.get(eid, np.empty((0, 2)))

        lane_blocked, min_dist, n_blockers = _lane_obstruction(
            bx, by, rx, ry, opponents
        )
        pass_dist = float(np.hypot(rx - bx, ry - by))
        pass_angle = _pass_angle_degrees(bx, by, rx, ry)

        results.append(
            {
                "event_id": eid,
                "receiver_player_id": row["receiver_player_id"],
                "lane_blocked": int(lane_blocked),
                "min_lane_defender_dist": min_dist,
                "defenders_in_corridor": n_blockers,
                "pass_distance": pass_dist,
                "pass_angle_deg": pass_angle,
            }
        )

    lane_df = pd.DataFrame(results)
    out = receivers_df.merge(
        lane_df, on=["event_id", "receiver_player_id"], how="left"
    )
    logger.info("Lane features computed for %d receiver candidates", len(out))
    return out


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _lane_obstruction(
    bx: float,
    by: float,
    rx: float,
    ry: float,
    opponents: np.ndarray,
    corridor_width: float = LANE_BLOCK_DISTANCE_M,
) -> tuple[bool, float, int]:
    """Determine lane obstruction between ball (bx, by) and receiver (rx, ry).

    Returns
    -------
    lane_blocked:
        True if any opponent is within ``corridor_width`` of the segment.
    min_lane_defender_dist:
        Minimum perpendicular distance from any opponent to the lane.
    defenders_in_corridor:
        Count of opponents within ``corridor_width`` of the segment.
    """
    if len(opponents) == 0:
        return False, np.inf, 0

    dists = np.array(
        [point_to_segment_distance(ox, oy, bx, by, rx, ry) for ox, oy in opponents]
    )
    min_dist = float(dists.min())
    n_blockers = int((dists <= corridor_width).sum())
    lane_blocked = n_blockers > 0
    return lane_blocked, min_dist, n_blockers


def point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Euclidean distance from point (px, py) to segment (ax, ay) → (bx, by).

    Clamps the projection parameter t to [0, 1] so the result is the
    distance to the *segment*, not the infinite line.
    """
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return float(np.hypot(px - ax, py - ay))
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return float(np.hypot(px - proj_x, py - proj_y))


def _pass_angle_degrees(bx: float, by: float, rx: float, ry: float) -> float:
    """Angle of the pass vector relative to the positive x-axis (forward).

    0° = directly forward (right), 90° = lateral, 180° = directly backward.
    Returned in [0, 180].
    """
    dx = rx - bx
    dy = ry - by
    angle_rad = np.arctan2(abs(dy), dx)
    return float(np.degrees(angle_rad))


def _standardise_frame_coords(frames: pd.DataFrame) -> pd.DataFrame:
    frames = frames.copy()
    flip = frames["_flip_frame"].fillna(False).astype(bool)
    frames["frame_x"] = np.where(flip, PITCH_LENGTH - frames["frame_x"], frames["frame_x"])
    frames["frame_y"] = np.where(flip, PITCH_WIDTH - frames["frame_y"], frames["frame_y"])
    return frames
