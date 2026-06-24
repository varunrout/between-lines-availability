"""
Phase 8 – Team- and player-level output metrics.

Computes football-facing metrics from the merged build-up / findability
dataset. These helpers are deliberately defensive: partial samples, missing
360 frames and empty candidate tables should still produce valid output
schemas instead of crashing the pipeline.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.features.findability import EVENT_SCORE_COLUMNS

logger = logging.getLogger(__name__)

EVENT_ZERO_COLUMNS = [
    "max_findability_score",
    "findable_option_available",
    "n_findable_candidates",
    "n_between_lines_candidates",
    "n_total_candidates",
    "central_findable_count",
    "half_space_findable_count",
]

TEAM_METRIC_COLUMNS = [
    "competition_name",
    "team",
    "total_buildup_events",
    "between_lines_availability_rate",
    "avg_findable_candidates",
    "avg_between_lines_candidates",
    "avg_line_gap",
    "central_access_rate",
    "half_space_access_rate",
    "missed_access_rate",
]

PLAYER_METRIC_COLUMNS = [
    "receiver_player_name",
    "times_between_lines",
    "times_findable",
    "receiver_availability_rate",
    "ball_carrier_recognition_rate",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_team_metrics(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Team-level between-lines availability metrics."""
    if merged_df.empty:
        return pd.DataFrame(columns=TEAM_METRIC_COLUMNS)

    df = _ensure_event_metric_columns(merged_df)
    group_cols = _get_group_cols(df, ["competition_name", "team"])
    if not group_cols:
        logger.warning("Cannot compute team metrics without grouping columns.")
        return pd.DataFrame(columns=TEAM_METRIC_COLUMNS)

    agg: dict[str, object] = {
        "total_buildup_events": ("findable_option_available", "count"),
        "between_lines_availability_rate": ("findable_option_available", "mean"),
        "avg_findable_candidates": ("n_findable_candidates", "mean"),
        "avg_between_lines_candidates": ("n_between_lines_candidates", "mean"),
    }

    if "line_gap_depth" in df.columns:
        agg["avg_line_gap"] = ("line_gap_depth", "mean")

    team_df = df.groupby(group_cols, dropna=False).agg(**agg).reset_index()

    total_findable = df.groupby(group_cols, dropna=False)["n_findable_candidates"].sum().replace(0, float("nan"))
    central = df.groupby(group_cols, dropna=False)["central_findable_count"].sum()
    half_space = df.groupby(group_cols, dropna=False)["half_space_findable_count"].sum()

    team_df = team_df.set_index(group_cols)
    team_df["central_access_rate"] = (central / total_findable).reindex(team_df.index)
    team_df["half_space_access_rate"] = (half_space / total_findable).reindex(team_df.index)

    if "pass_to_between_lines" in df.columns:
        missed = df[df["findable_option_available"] == 1].copy()
        if not missed.empty:
            missed_rate = (1 - missed.groupby(group_cols, dropna=False)["pass_to_between_lines"].mean()).rename(
                "missed_access_rate"
            )
            team_df = team_df.join(missed_rate.reindex(team_df.index))

    team_df = team_df.reset_index()
    team_df["between_lines_availability_rate"] = (team_df["between_lines_availability_rate"] * 100).round(1)

    for col in TEAM_METRIC_COLUMNS:
        if col not in team_df.columns:
            team_df[col] = pd.NA

    team_df = team_df.sort_values("between_lines_availability_rate", ascending=False)
    logger.info("Team metrics computed for %d teams", len(team_df))
    return team_df[TEAM_METRIC_COLUMNS].reset_index(drop=True)


def compute_player_metrics(
    merged_df: pd.DataFrame,
    receivers_scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """Player-level between-lines availability metrics."""
    from src.config import FINDABLE_SCORE_THRESHOLD

    if receivers_scored_df.empty or "between_lines" not in receivers_scored_df.columns:
        return pd.DataFrame(columns=PLAYER_METRIC_COLUMNS)

    rec = receivers_scored_df[receivers_scored_df["between_lines"].fillna(False)].copy()
    if rec.empty:
        return pd.DataFrame(columns=PLAYER_METRIC_COLUMNS)

    rec["is_findable"] = rec["findability_score"] >= FINDABLE_SCORE_THRESHOLD

    receiver_agg = (
        rec.groupby("receiver_player_name", dropna=False)
        .agg(
            times_between_lines=("between_lines", "sum"),
            times_findable=("is_findable", "sum"),
        )
        .reset_index()
    )
    receiver_agg["receiver_availability_rate"] = (
        receiver_agg["times_findable"] / receiver_agg["times_between_lines"].replace(0, float("nan"))
    )

    if "player" in merged_df.columns and "pass_to_between_lines" in merged_df.columns:
        merged_for_recognition = _ensure_event_metric_columns(merged_df)
        avail_events = merged_for_recognition[merged_for_recognition["findable_option_available"] == 1]
        if not avail_events.empty:
            recognition = (
                avail_events.groupby("player", dropna=False)["pass_to_between_lines"]
                .mean()
                .rename("ball_carrier_recognition_rate")
                .reset_index()
                .rename(columns={"player": "receiver_player_name"})
            )
            receiver_agg = receiver_agg.merge(recognition, on="receiver_player_name", how="left")

    for col in PLAYER_METRIC_COLUMNS:
        if col not in receiver_agg.columns:
            receiver_agg[col] = pd.NA

    receiver_agg = receiver_agg.sort_values("times_findable", ascending=False)
    logger.info("Player metrics computed for %d players", len(receiver_agg))
    return receiver_agg[PLAYER_METRIC_COLUMNS].reset_index(drop=True)


def merge_event_scores(
    buildup_df: pd.DataFrame,
    event_scores_df: pd.DataFrame,
    line_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join build-up events with event-level findability scores and line data."""
    event_scores = _normalise_event_scores(event_scores_df)

    merged = buildup_df.merge(
        event_scores,
        left_on="id",
        right_on="event_id",
        how="left",
    )

    if line_df.empty or "event_id" not in line_df.columns:
        line_df = pd.DataFrame(columns=["event_id", "defensive_line_x", "midfield_line_x", "line_gap_depth"])

    merged = merged.merge(
        line_df,
        left_on="id",
        right_on="event_id",
        how="left",
        suffixes=("", "_line"),
    )

    merged = _ensure_event_metric_columns(merged)
    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_event_scores(event_scores_df: pd.DataFrame) -> pd.DataFrame:
    """Return event scores with the expected schema, even when empty."""
    if event_scores_df.empty or "event_id" not in event_scores_df.columns:
        return pd.DataFrame(columns=EVENT_SCORE_COLUMNS)

    scores = event_scores_df.copy()
    for col in EVENT_SCORE_COLUMNS:
        if col not in scores.columns:
            scores[col] = pd.NA
    return scores[EVENT_SCORE_COLUMNS]


def _ensure_event_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing event-level scoring fields with safe defaults."""
    out = df.copy()
    for col in EVENT_ZERO_COLUMNS:
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0).astype(int)
    return out


def _get_group_cols(df: pd.DataFrame, preferred: list[str]) -> list[str]:
    """Return those preferred grouping columns that actually exist in df."""
    return [c for c in preferred if c in df.columns]
