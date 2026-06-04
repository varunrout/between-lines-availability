"""
Phase 9 – Visualisation helpers.

Provides all pitch and chart plots for the between-lines availability project.

Functions
---------
plot_freeze_frame
    Single-event freeze-frame pitch plot showing ball-carrier, receiver
    candidates, passing lanes and between-lines zone.

plot_team_heatmap
    Spatial heatmap of where a team creates between-lines options.

plot_availability_scatter
    Scatter of receiver availability vs usage rate (recruitment view).

plot_team_comparison_bar
    Horizontal bar chart of team-level availability rates.

plot_missed_opportunity
    Freeze-frame for an event where a findable option existed but was not used.

Usage
-----
    from src.viz.plots import plot_freeze_frame, plot_team_comparison_bar
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from mplsoccer import Pitch, VerticalPitch
    MPLSOCCER_AVAILABLE = True
except ImportError:
    MPLSOCCER_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from src.config import (
    CENTRAL_Y_MAX,
    CENTRAL_Y_MIN,
    FINDABLE_SCORE_THRESHOLD,
    HALF_SPACE_HIGH_Y_MAX,
    HALF_SPACE_HIGH_Y_MIN,
    HALF_SPACE_LOW_Y_MAX,
    HALF_SPACE_LOW_Y_MIN,
    PITCH_LENGTH,
    PITCH_WIDTH,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOURS = {
    "ball_carrier": "#FFD700",
    "findable_receiver": "#00CC44",
    "unfindable_receiver": "#FF6600",
    "opponent": "#CC0000",
    "lane_clear": "#00CC44",
    "lane_blocked": "#CC0000",
    "between_lines_zone": "#ADD8E6",
    "defensive_line": "#CC0000",
    "midfield_line": "#FF6600",
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def plot_freeze_frame(
    event_id: str,
    frames_df: pd.DataFrame,
    buildup_df: pd.DataFrame,
    line_df: pd.DataFrame,
    candidates_scored_df: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Render a single freeze-frame event on a StatsBomb pitch.

    Shows:
    - Ball carrier position (gold star)
    - Opponent players (red dots)
    - Between-lines zone (shaded rectangle)
    - Defensive line (red dashed line)
    - Midfield line (orange dashed line)
    - Receiver candidates: green (findable) or orange (not findable)
    - Passing lanes: green (clear) or red (blocked)

    Parameters
    ----------
    event_id:
        UUID of the event to plot.
    frames_df / buildup_df / line_df / candidates_scored_df:
        The pipeline DataFrames containing this event's data.
    """
    if not MPLSOCCER_AVAILABLE:
        raise ImportError("mplsoccer is required for pitch plots.")

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="white")
    fig, ax = pitch.draw(figsize=(12, 8))

    # --- Ball carrier ---
    event_row = buildup_df[buildup_df["id"] == event_id]
    if event_row.empty:
        logger.warning("Event %s not found in buildup_df", event_id)
        return fig
    bx = float(event_row["x"].iloc[0])
    by = float(event_row["y"].iloc[0])

    ax.scatter(bx, by, s=300, marker="*", color=COLOURS["ball_carrier"],
               zorder=6, label="Ball carrier", edgecolor="black", linewidth=0.5)

    # --- Opponent lines ---
    lines = line_df[line_df["event_id"] == event_id]
    if not lines.empty:
        def_x = float(lines["defensive_line_x"].iloc[0])
        mid_x = float(lines["midfield_line_x"].iloc[0])

        # Between-lines zone shading
        zone_width = def_x - mid_x
        if zone_width > 0:
            rect = mpatches.Rectangle(
                (mid_x, 0), zone_width, PITCH_WIDTH,
                linewidth=0, facecolor=COLOURS["between_lines_zone"], alpha=0.25,
                zorder=1, label="Between-lines zone",
            )
            ax.add_patch(rect)

        ax.axvline(def_x, color=COLOURS["defensive_line"], linestyle="--",
                   linewidth=2, alpha=0.8, label=f"Defensive line (x={def_x:.0f})")
        ax.axvline(mid_x, color=COLOURS["midfield_line"], linestyle="--",
                   linewidth=2, alpha=0.8, label=f"Midfield line (x={mid_x:.0f})")

    # --- Opponent players ---
    flip_info = buildup_df[buildup_df["id"] == event_id][["id", "_flip_frame"]]
    flip_info = flip_info.rename(columns={"id": "event_id"})
    frame_players = frames_df[frames_df["event_id"] == event_id].copy()
    if not flip_info.empty and not frame_players.empty:
        flip = bool(flip_info["_flip_frame"].iloc[0])
        if flip:
            frame_players["frame_x"] = PITCH_LENGTH - frame_players["frame_x"]
            frame_players["frame_y"] = PITCH_WIDTH - frame_players["frame_y"]

        opponents = frame_players[~frame_players["teammate"]]
        if not opponents.empty:
            ax.scatter(
                opponents["frame_x"], opponents["frame_y"],
                s=120, color=COLOURS["opponent"], zorder=4,
                label="Opponent", edgecolor="white", linewidth=0.5,
            )

    # --- Receiver candidates ---
    cands = candidates_scored_df[candidates_scored_df["event_id"] == event_id]
    for _, cand in cands.iterrows():
        rx, ry = cand["receiver_x"], cand["receiver_y"]
        is_findable = cand.get("findability_score", 0) >= FINDABLE_SCORE_THRESHOLD
        color = COLOURS["findable_receiver"] if is_findable else COLOURS["unfindable_receiver"]
        lane_col = COLOURS["lane_clear"] if not cand.get("lane_blocked", 1) else COLOURS["lane_blocked"]

        ax.scatter(rx, ry, s=150, color=color, zorder=5,
                   edgecolor="white", linewidth=0.5)

        # Draw passing lane
        ax.plot([bx, rx], [by, ry], color=lane_col, linewidth=1.5,
                linestyle="--" if cand.get("lane_blocked", 1) else "-", alpha=0.7, zorder=3)

    # --- Title and legend ---
    player_name = event_row.get("player", pd.Series([event_id])).iloc[0]
    event_type = event_row.get("type", pd.Series(["Event"])).iloc[0]
    ax.set_title(
        title or f"{player_name} – {event_type} (event {event_id[:8]}…)",
        color="white", fontsize=13, pad=10,
    )

    legend = ax.legend(
        loc="upper left", fontsize=8, facecolor="#1a1a2e",
        edgecolor="white", labelcolor="white",
    )
    ax.add_artist(legend)
    _add_green_unfindable_legend(ax)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info("Freeze-frame saved to %s", save_path)

    return fig


def plot_team_heatmap(
    merged_df: pd.DataFrame,
    team: str,
    competition: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Pitch heatmap showing where a team creates findable between-lines options.

    Parameters
    ----------
    merged_df:
        Merged build-up + findability DataFrame.
    team:
        Team name to filter.
    """
    if not MPLSOCCER_AVAILABLE:
        raise ImportError("mplsoccer is required for pitch plots.")

    subset = merged_df[merged_df["team"] == team].copy()
    if competition:
        subset = subset[subset["competition_name"] == competition]

    findable = subset[subset["findable_option_available"] == 1]

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="white")
    fig, ax = pitch.draw(figsize=(12, 8))

    if not findable.empty:
        bin_statistic = pitch.bin_statistic(
            findable["x"], findable["y"],
            statistic="count", bins=(24, 16),
        )
        pitch.heatmap(bin_statistic, ax=ax, cmap="YlOrRd", edgecolors="#1a1a2e")

    title = f"{team} – Findable Between-Lines Options"
    if competition:
        title += f" ({competition})"
    ax.set_title(title, color="white", fontsize=13, pad=10)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def plot_availability_scatter(
    player_metrics_df: pd.DataFrame,
    min_appearances: int = 20,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Scatter plot: receiver availability rate vs times findable.

    Parameters
    ----------
    player_metrics_df:
        Output of ``src.metrics.outputs.compute_player_metrics``.
    min_appearances:
        Minimum ``times_between_lines`` to include in chart.
    """
    df = player_metrics_df[
        player_metrics_df["times_between_lines"] >= min_appearances
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    scatter = ax.scatter(
        df["times_findable"],
        df["receiver_availability_rate"] * 100,
        s=80,
        c=df["times_between_lines"],
        cmap="YlOrRd",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.9,
    )

    # Label top players
    top = df.nlargest(10, "times_findable")
    for _, row in top.iterrows():
        ax.annotate(
            row["receiver_player_name"],
            (row["times_findable"], row["receiver_availability_rate"] * 100),
            color="white", fontsize=7, ha="left",
            xytext=(4, 2), textcoords="offset points",
        )

    plt.colorbar(scatter, ax=ax, label="Times between lines").ax.yaxis.label.set_color("white")

    ax.set_xlabel("Times Findable", color="white")
    ax.set_ylabel("Receiver Availability Rate (%)", color="white")
    ax.set_title("Receiver Availability – Volume vs Rate", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def plot_team_comparison_bar(
    team_metrics_df: pd.DataFrame,
    metric: str = "between_lines_availability_rate",
    top_n: int = 20,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Horizontal bar chart comparing teams on a selected metric.

    Parameters
    ----------
    team_metrics_df:
        Output of ``src.metrics.outputs.compute_team_metrics``.
    metric:
        Column name to plot.
    top_n:
        Number of teams to show.
    """
    df = team_metrics_df.dropna(subset=[metric]).nlargest(top_n, metric)

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.4)))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    bars = ax.barh(df["team"], df[metric], color="#00CC44", edgecolor="#1a1a2e")

    ax.set_xlabel(
        metric.replace("_", " ").title() + (" (%)" if "rate" in metric else ""),
        color="white",
    )
    ax.set_title(f"Team Comparison – {metric.replace('_', ' ').title()}", color="white")
    ax.tick_params(colors="white")
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    # Value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}", va="center", ha="left", color="white", fontsize=8)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def plot_missed_opportunity(
    event_id: str,
    frames_df: pd.DataFrame,
    buildup_df: pd.DataFrame,
    line_df: pd.DataFrame,
    candidates_scored_df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Freeze-frame highlighting a missed between-lines opportunity.

    Annotates the best findable receiver that was not played to,
    with a highlighted glow effect.
    """
    fig = plot_freeze_frame(
        event_id=event_id,
        frames_df=frames_df,
        buildup_df=buildup_df,
        line_df=line_df,
        candidates_scored_df=candidates_scored_df,
        title="⚠️  Missed Between-Lines Opportunity",
    )

    # Add a "missed" annotation on the best findable receiver
    cands = candidates_scored_df[candidates_scored_df["event_id"] == event_id]
    findable = cands[cands.get("findability_score", pd.Series(dtype=float)) >= FINDABLE_SCORE_THRESHOLD]
    if not findable.empty and len(fig.axes) > 0:
        if "findability_score" in findable.columns:
            best = findable.sort_values("findability_score", ascending=False).iloc[0]
        else:
            best = findable.iloc[0]
        ax = fig.axes[0]
        ax.scatter(
            best["receiver_x"], best["receiver_y"],
            s=500, color="none", edgecolor="yellow",
            linewidth=2.5, zorder=7,
        )
        ax.annotate(
            "← Best option\n   not played",
            (best["receiver_x"], best["receiver_y"]),
            color="yellow", fontsize=9, fontweight="bold",
            xytext=(8, 8), textcoords="offset points",
        )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_green_unfindable_legend(ax: plt.Axes) -> None:
    handles = [
        mpatches.Patch(color=COLOURS["findable_receiver"], label="Findable receiver"),
        mpatches.Patch(color=COLOURS["unfindable_receiver"], label="Not findable"),
    ]
    ax.legend(
        handles=handles, loc="upper right", fontsize=8,
        facecolor="#1a1a2e", edgecolor="white", labelcolor="white",
    )
