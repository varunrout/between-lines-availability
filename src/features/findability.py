"""
Phase 6 – Rule-based findability scoring.

Aggregates per-candidate features into a findability score (0–3) and then
collapses to an event-level binary flag and summary statistics.

Scoring rules (per candidate)
-----------------------------
  Base: candidate must be between_lines, ahead_of_ball, visible, and central/half-space
  +1 : pass_distance < MAX_PASS_DISTANCE_M          (reachable)
  +1 : nearest_defender_dist > MIN_RECEIVER_SPACE_M (not tightly marked)
  +1 : lane_blocked == 0                            (clear lane)

Event-level aggregation
-----------------------
  max_findability_score      : max score among all candidates at this event
  n_findable_candidates      : count with score >= FINDABLE_SCORE_THRESHOLD
  findable_option_available  : 1 if any candidate has score >= threshold
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import (
    FINDABLE_SCORE_THRESHOLD,
    MAX_PASS_DISTANCE_M,
    MIN_RECEIVER_SPACE_M,
)

logger = logging.getLogger(__name__)

VALID_RECEIVER_ZONES = {"central", "half_space_low", "half_space_high"}

EVENT_SCORE_COLUMNS = [
    "event_id",
    "max_findability_score",
    "findable_option_available",
    "n_findable_candidates",
    "n_between_lines_candidates",
    "n_total_candidates",
    "central_findable_count",
    "half_space_findable_count",
    "best_receiver_x",
    "best_receiver_y",
    "best_receiver_name",
    "best_pass_distance",
    "best_nearest_defender",
    "best_lane_blocked",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_findability(candidates_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each receiver candidate and aggregate to event level.

    Parameters
    ----------
    candidates_df:
        Output of ``src.features.lane.compute_lane_features``. Must have
        candidate-level geometry, pressure and lane features.

    Returns
    -------
    event_scores : pd.DataFrame
        One row per event_id with event-level findability summary.
    candidates_scored : pd.DataFrame
        All candidates with an added ``findability_score`` column.
    """
    if candidates_df.empty:
        scored = candidates_df.copy()
        if "findability_score" not in scored.columns:
            scored["findability_score"] = pd.Series(dtype="int64")
        return pd.DataFrame(columns=EVENT_SCORE_COLUMNS), scored

    df = candidates_df.copy()
    df["findability_score"] = df.apply(_score_candidate, axis=1).astype(int)

    event_agg = (
        df.groupby("event_id", group_keys=False)
        .apply(_aggregate_event)
        .reset_index()
    )
    event_agg = event_agg.reindex(columns=EVENT_SCORE_COLUMNS)

    availability_rate = 0.0 if event_agg.empty else 100.0 * event_agg["findable_option_available"].mean()
    logger.info(
        "Findability: %d events, %.1f%% have at least one findable option",
        len(event_agg),
        availability_rate,
    )
    return event_agg, df


def score_candidate(row: pd.Series) -> int:
    """Public single-row scoring helper, also used by tests."""
    return _score_candidate(row)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_candidate(row: pd.Series) -> int:
    """Compute findability score from rule-based candidate features."""
    if not bool(row.get("between_lines", False)):
        return 0
    if not bool(row.get("ahead_of_ball", False)):
        return 0

    if "in_visible_area" in row.index and not bool(row["in_visible_area"]):
        return 0

    centrality_zone = row.get("centrality_zone", "wide")
    if pd.notna(centrality_zone) and centrality_zone not in VALID_RECEIVER_ZONES:
        return 0

    score = 0

    pass_dist = row.get("pass_distance", float("inf"))
    if pd.notna(pass_dist) and float(pass_dist) < MAX_PASS_DISTANCE_M:
        score += 1

    nearest = row.get("nearest_defender_dist", 0.0)
    if pd.notna(nearest) and float(nearest) > MIN_RECEIVER_SPACE_M:
        score += 1

    lane_blocked = row.get("lane_blocked", 1)
    if pd.notna(lane_blocked) and int(lane_blocked) == 0:
        score += 1

    return score


def _aggregate_event(group: pd.DataFrame) -> pd.Series:
    """Collapse candidate rows to a single event-level summary."""
    max_score = int(group["findability_score"].max())
    n_findable = int((group["findability_score"] >= FINDABLE_SCORE_THRESHOLD).sum())
    n_between_lines = int(group["between_lines"].sum())
    n_candidates = int(len(group))

    findable = group[group["findability_score"] >= FINDABLE_SCORE_THRESHOLD]
    central_findable = int((findable["centrality_zone"] == "central").sum())
    half_space_findable = int(
        findable["centrality_zone"].isin(["half_space_low", "half_space_high"]).sum()
    )

    best_idx = group["findability_score"].idxmax() if not group.empty else None
    best_candidate = group.loc[best_idx] if best_idx is not None else pd.Series(dtype=object)

    return pd.Series(
        {
            "max_findability_score": max_score,
            "findable_option_available": int(max_score >= FINDABLE_SCORE_THRESHOLD),
            "n_findable_candidates": n_findable,
            "n_between_lines_candidates": n_between_lines,
            "n_total_candidates": n_candidates,
            "central_findable_count": central_findable,
            "half_space_findable_count": half_space_findable,
            "best_receiver_x": best_candidate.get("receiver_x"),
            "best_receiver_y": best_candidate.get("receiver_y"),
            "best_receiver_name": best_candidate.get("receiver_player_name"),
            "best_pass_distance": best_candidate.get("pass_distance"),
            "best_nearest_defender": best_candidate.get("nearest_defender_dist"),
            "best_lane_blocked": best_candidate.get("lane_blocked"),
        }
    )
