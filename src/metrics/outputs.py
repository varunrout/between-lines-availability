"""
Phase 8 – Team- and player-level output metrics.

Computes the key football-facing metrics from the merged build-up /
findability dataset.

Usage
-----
    from src.metrics.outputs import compute_team_metrics, compute_player_metrics

    team_df = compute_team_metrics(merged_df)
    player_df = compute_player_metrics(merged_df, receivers_scored_df)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_team_metrics(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Team-level between-lines availability metrics.

    Parameters
    ----------
    merged_df:
        Merged DataFrame combining build-up events with event-level findability
        scores.  Must contain: ``team``, ``competition_name``,
        ``findable_option_available``, ``n_findable_candidates``,
        ``n_between_lines_candidates``, ``central_findable_count``,
        ``half_space_findable_count``, and optionally
        ``pass_to_between_lines`` for missed-access calculation.

    Returns
    -------
    pd.DataFrame
        One row per (competition, team) with the following metrics:

        between_lines_availability_rate
            % of build-up events with at least one findable between-lines option.
        central_access_rate
            % of findable options that are in a central zone.
        half_space_access_rate
            % of findable options in a half-space zone.
        missed_access_rate
            % of events where a findable option existed but was NOT used
            (requires ``pass_to_between_lines`` column; NaN otherwise).
        avg_line_gap
            Mean gap between midfield and defensive lines.
        total_buildup_events
            Total build-up events analysed.
    """
    group_cols = _get_group_cols(merged_df, ["competition_name", "team"])

    agg: dict[str, object] = {
        "total_buildup_events": ("findable_option_available", "count"),
        "between_lines_availability_rate": ("findable_option_available", "mean"),
        "avg_findable_candidates": ("n_findable_candidates", "mean"),
        "avg_between_lines_candidates": ("n_between_lines_candidates", "mean"),
    }

    if "line_gap_depth" in merged_df.columns:
        agg["avg_line_gap"] = ("line_gap_depth", "mean")

    team_df = merged_df.groupby(group_cols).agg(**agg).reset_index()

    # Central & half-space access rates
    total_findable = merged_df.groupby(group_cols)["n_findable_candidates"].sum()
    central = merged_df.groupby(group_cols)["central_findable_count"].sum()
    half_space = merged_df.groupby(group_cols)["half_space_findable_count"].sum()
    total_findable = total_findable.replace(0, float("nan"))

    team_df = team_df.set_index(group_cols)
    team_df["central_access_rate"] = (central / total_findable).reindex(team_df.index)
    team_df["half_space_access_rate"] = (half_space / total_findable).reindex(team_df.index)

    # Missed access rate (optional – requires pass_to_between_lines column)
    if "pass_to_between_lines" in merged_df.columns:
        missed = merged_df[merged_df["findable_option_available"] == 1].copy()
        missed_rate = (
            1 - missed.groupby(group_cols)["pass_to_between_lines"].mean()
        ).rename("missed_access_rate")
        team_df = team_df.join(
            missed_rate.reindex(team_df.index)
        )

    team_df = team_df.reset_index()
    team_df["between_lines_availability_rate"] = (
        team_df["between_lines_availability_rate"] * 100
    ).round(1)

    team_df = team_df.sort_values("between_lines_availability_rate", ascending=False)
    logger.info("Team metrics computed for %d teams", len(team_df))
    return team_df.reset_index(drop=True)


def compute_player_metrics(
    merged_df: pd.DataFrame,
    receivers_scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """Player-level between-lines availability metrics.

    Parameters
    ----------
    merged_df:
        Merged build-up + findability events with ``team`` and
        optionally ``player`` (ball-carrier name).
    receivers_scored_df:
        Scored receiver candidates from
        ``src.features.findability.compute_findability``.
        Must have: ``receiver_player_name``, ``findability_score``,
        ``between_lines``, ``event_id``.

    Returns
    -------
    pd.DataFrame
        One row per player with:

        receiver_availability_rate
            % of their appearances between the lines that are findable.
        times_between_lines
            Total events where they appeared between the lines.
        times_findable
            Total events where they were findable (score ≥ threshold).
        ball_carrier_recognition_rate
            (ball-carrier metric) % of events where player found the
            between-lines option when available.  Requires
            ``pass_to_between_lines`` in ``merged_df``.
    """
    from src.config import FINDABLE_SCORE_THRESHOLD

    # --- Receiver metrics ---
    rec = receivers_scored_df[receivers_scored_df["between_lines"]].copy()
    rec["is_findable"] = rec["findability_score"] >= FINDABLE_SCORE_THRESHOLD

    receiver_agg = (
        rec.groupby("receiver_player_name")
        .agg(
            times_between_lines=("between_lines", "sum"),
            times_findable=("is_findable", "sum"),
        )
        .reset_index()
    )
    receiver_agg["receiver_availability_rate"] = (
        receiver_agg["times_findable"] / receiver_agg["times_between_lines"].replace(0, float("nan"))
    )

    # --- Ball-carrier recognition (optional) ---
    if "player" in merged_df.columns and "pass_to_between_lines" in merged_df.columns:
        avail_events = merged_df[merged_df["findable_option_available"] == 1]
        recognition = (
            avail_events.groupby("player")["pass_to_between_lines"]
            .mean()
            .rename("ball_carrier_recognition_rate")
            .reset_index()
            .rename(columns={"player": "receiver_player_name"})
        )
        receiver_agg = receiver_agg.merge(recognition, on="receiver_player_name", how="left")

    receiver_agg = receiver_agg.sort_values("times_findable", ascending=False)
    logger.info("Player metrics computed for %d players", len(receiver_agg))
    return receiver_agg.reset_index(drop=True)


def merge_event_scores(
    buildup_df: pd.DataFrame,
    event_scores_df: pd.DataFrame,
    line_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join build-up events with event-level findability scores and line data.

    This produces the primary analysis DataFrame used by metrics and
    visualisations.
    """
    merged = buildup_df.merge(
        event_scores_df, left_on="id", right_on="event_id", how="left"
    )
    merged = merged.merge(
        line_df, left_on="id", right_on="event_id", how="left", suffixes=("", "_line")
    )
    # Fill events with no 360 data
    for col in ["findable_option_available", "max_findability_score"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_group_cols(df: pd.DataFrame, preferred: list[str]) -> list[str]:
    """Return those preferred grouping columns that actually exist in df."""
    return [c for c in preferred if c in df.columns]
