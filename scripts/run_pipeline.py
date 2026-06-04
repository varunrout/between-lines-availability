"""
End-to-end pipeline runner for the Between-Lines Availability Model.

Executes all phases in sequence for all configured competitions and
writes output artefacts to the ``outputs/`` directory.

Usage
-----
    python scripts/run_pipeline.py [--competitions euro2020 wc2022 euro2024]
                                   [--output-dir outputs]
                                   [--skip-model]
                                   [--max-matches N]

Output artefacts
----------------
outputs/
  buildup_events.csv          – filtered build-up events with direction std
  line_features.csv           – per-event opponent line metrics
  receiver_candidates.csv     – per-candidate receiver features + lane
  event_scores.csv            – per-event findability aggregation
  merged_analysis.csv         – full joined analysis table
  team_metrics.csv            – team-level summary metrics
  player_metrics.csv          – player-level summary metrics
  model/availability_model.pkl
  plots/
    team_comparison.png
    availability_scatter.png
    calibration_curves.png
    freeze_frames/
      example_findable_<N>.png
      missed_opportunity_<N>.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import COMPETITIONS
from src.data.loader import load_competition
from src.features.buildup import filter_buildup
from src.features.findability import compute_findability
from src.features.lane import compute_lane_features
from src.features.line_detection import detect_opponent_lines
from src.features.receiver import detect_receiver_candidates
from src.metrics.outputs import (
    compute_player_metrics,
    compute_team_metrics,
    merge_event_scores,
)
from src.models.availability import AvailabilityModel
from src.viz.plots import (
    plot_freeze_frame,
    plot_missed_opportunity,
    plot_team_comparison_bar,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

COMP_SHORTCUTS = {
    "euro2020": {"competition_id": 55, "season_id": 43,  "name": "UEFA Euro 2020"},
    "wc2022":   {"competition_id": 43, "season_id": 106, "name": "FIFA World Cup 2022"},
    "euro2024": {"competition_id": 55, "season_id": 282, "name": "UEFA Euro 2024"},
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Between-Lines Availability Pipeline")
    p.add_argument(
        "--competitions",
        nargs="+",
        default=["euro2020", "wc2022", "euro2024"],
        choices=list(COMP_SHORTCUTS.keys()),
        help="Competitions to include (default: all three).",
    )
    p.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write all output artefacts (default: outputs/).",
    )
    p.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip the ML modelling phase.",
    )
    p.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Limit matches per competition (useful for quick tests).",
    )
    p.add_argument(
        "--n-freeze-frame-examples",
        type=int,
        default=5,
        help="Number of example freeze-frame plots to save.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots" / "freeze_frames").mkdir(parents=True, exist_ok=True)

    competitions = [COMP_SHORTCUTS[c] for c in args.competitions]

    all_events, all_frames, all_lineups, all_matches = [], [], [], []

    # ---------------------------------------------------------------
    # Phase 1: Data loading
    # ---------------------------------------------------------------
    for comp in competitions:
        logger.info("=== Loading: %s ===", comp["name"])
        ev, fr, lu, ma = load_competition(
            competition_id=comp["competition_id"],
            season_id=comp["season_id"],
        )
        ev["competition_name"] = comp["name"]
        if not fr.empty:
            fr["competition_name"] = comp["name"]
        if not lu.empty:
            lu["competition_name"] = comp["name"]
        ma["competition_name"] = comp["name"]

        if args.max_matches:
            allowed_ids = set(ma["match_id"].head(args.max_matches))
            ev = ev[ev["match_id"].isin(allowed_ids)]
            fr = fr[fr["match_id"].isin(allowed_ids)] if not fr.empty else fr

        all_events.append(ev)
        all_frames.append(fr)
        all_lineups.append(lu)
        all_matches.append(ma)

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    non_empty_frames = [f for f in all_frames if not f.empty]
    frames_df = pd.concat(non_empty_frames, ignore_index=True) if non_empty_frames else pd.DataFrame()
    non_empty_lineups = [l for l in all_lineups if not l.empty]
    lineups_df = pd.concat(non_empty_lineups, ignore_index=True) if non_empty_lineups else pd.DataFrame()
    matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()

    logger.info(
        "Total: %d events, %d frame rows across %d matches",
        len(events_df), len(frames_df), len(matches_df),
    )

    # ---------------------------------------------------------------
    # Phase 2: Build-up filtering
    # ---------------------------------------------------------------
    logger.info("=== Phase 2: Build-up filtering ===")
    buildup_df = filter_buildup(events_df)
    buildup_df.to_csv(out_dir / "buildup_events.csv", index=False)
    logger.info("Saved buildup_events.csv (%d rows)", len(buildup_df))

    # ---------------------------------------------------------------
    # Phase 3: Opponent line detection
    # ---------------------------------------------------------------
    logger.info("=== Phase 3: Opponent line detection ===")
    line_df = detect_opponent_lines(frames_df, buildup_df)
    if not line_df.empty:
        line_df.to_csv(out_dir / "line_features.csv", index=False)
        logger.info("Saved line_features.csv (%d rows)", len(line_df))

    # ---------------------------------------------------------------
    # Phase 4: Receiver candidates
    # ---------------------------------------------------------------
    logger.info("=== Phase 4: Receiver candidate detection ===")
    receivers_df = detect_receiver_candidates(frames_df, buildup_df, line_df)

    # ---------------------------------------------------------------
    # Phase 5: Passing lane features
    # ---------------------------------------------------------------
    logger.info("=== Phase 5: Passing lane obstruction ===")
    candidates_df = compute_lane_features(receivers_df, frames_df, buildup_df)
    if not candidates_df.empty:
        candidates_df.to_csv(out_dir / "receiver_candidates.csv", index=False)
        logger.info("Saved receiver_candidates.csv (%d rows)", len(candidates_df))

    # ---------------------------------------------------------------
    # Phase 6: Findability scoring
    # ---------------------------------------------------------------
    logger.info("=== Phase 6: Findability scoring ===")
    event_scores, candidates_scored = compute_findability(candidates_df)
    if not event_scores.empty:
        event_scores.to_csv(out_dir / "event_scores.csv", index=False)
        logger.info("Saved event_scores.csv (%d rows)", len(event_scores))

    # ---------------------------------------------------------------
    # Merge
    # ---------------------------------------------------------------
    logger.info("=== Merging analysis table ===")
    merged_df = merge_event_scores(buildup_df, event_scores, line_df)
    merged_df.to_csv(out_dir / "merged_analysis.csv", index=False)
    logger.info("Saved merged_analysis.csv (%d rows)", len(merged_df))

    # ---------------------------------------------------------------
    # Phase 8: Output metrics
    # ---------------------------------------------------------------
    logger.info("=== Phase 8: Computing metrics ===")
    team_metrics = compute_team_metrics(merged_df)
    team_metrics.to_csv(out_dir / "team_metrics.csv", index=False)

    player_metrics = compute_player_metrics(merged_df, candidates_scored)
    player_metrics.to_csv(out_dir / "player_metrics.csv", index=False)

    logger.info("Team metrics:\n%s", team_metrics[
        ["team", "between_lines_availability_rate", "total_buildup_events"]
    ].head(10).to_string(index=False))

    # ---------------------------------------------------------------
    # Phase 7: ML model (optional)
    # ---------------------------------------------------------------
    if not args.skip_model and not merged_df.empty:
        logger.info("=== Phase 7: ML Availability Classifier ===")
        model_df = merged_df.dropna(subset=["findable_option_available"])
        if len(model_df) >= 100:
            model = AvailabilityModel()
            model.fit(model_df)
            metrics = model.evaluate(model_df, plot=True)
            logger.info("Model metrics: %s", metrics)
            model_dir = out_dir / "model"
            model_dir.mkdir(exist_ok=True)
            model.save(model_dir / "availability_model.pkl")
        else:
            logger.warning("Not enough labelled data for model training (%d rows).", len(model_df))

    # ---------------------------------------------------------------
    # Phase 9: Visualisations
    # ---------------------------------------------------------------
    logger.info("=== Phase 9: Visualisations ===")
    try:
        if not team_metrics.empty:
            plot_team_comparison_bar(
                team_metrics,
                save_path=out_dir / "plots" / "team_comparison.png",
            )

        if not player_metrics.empty:
            from src.viz.plots import plot_availability_scatter
            plot_availability_scatter(
                player_metrics,
                save_path=out_dir / "plots" / "availability_scatter.png",
            )

        # Sample freeze-frame examples
        _save_freeze_frame_examples(
            merged_df=merged_df,
            frames_df=frames_df,
            buildup_df=buildup_df,
            line_df=line_df,
            candidates_scored_df=candidates_scored,
            out_dir=out_dir,
            n=args.n_freeze_frame_examples,
        )
    except Exception as exc:
        logger.warning("Visualisation error (non-fatal): %s", exc)

    logger.info("=== Pipeline complete. Outputs in %s ===", out_dir.resolve())


# ---------------------------------------------------------------------------
# Freeze-frame helper
# ---------------------------------------------------------------------------

def _save_freeze_frame_examples(
    merged_df,
    frames_df,
    buildup_df,
    line_df,
    candidates_scored_df,
    out_dir: Path,
    n: int = 5,
) -> None:
    """Save N freeze-frame examples and N missed-opportunity examples."""
    ff_dir = out_dir / "plots" / "freeze_frames"

    # Findable events
    findable_ids = merged_df[merged_df["findable_option_available"] == 1]["id"].dropna()
    events_with_frames = set(frames_df["event_id"].unique())
    findable_ids = [eid for eid in findable_ids if eid in events_with_frames]

    for i, eid in enumerate(findable_ids[:n]):
        try:
            plot_freeze_frame(
                event_id=eid,
                frames_df=frames_df,
                buildup_df=buildup_df,
                line_df=line_df,
                candidates_scored_df=candidates_scored_df,
                save_path=ff_dir / f"example_findable_{i+1}.png",
            )
        except Exception as exc:
            logger.debug("Skipped freeze-frame %s: %s", eid, exc)

    # Missed opportunities
    missed_ids: list = []
    if "pass_to_between_lines" in merged_df.columns:
        missed_ids = merged_df[
            (merged_df["findable_option_available"] == 1) &
            (merged_df["pass_to_between_lines"] == 0)
        ]["id"].dropna().tolist()
        missed_ids = [eid for eid in missed_ids if eid in events_with_frames]
    else:
        logger.debug(
            "Skipping missed-opportunity plots: pass_to_between_lines column not available."
        )

    for i, eid in enumerate(missed_ids[:n]):
        try:
            plot_missed_opportunity(
                event_id=eid,
                frames_df=frames_df,
                buildup_df=buildup_df,
                line_df=line_df,
                candidates_scored_df=candidates_scored_df,
                save_path=ff_dir / f"missed_opportunity_{i+1}.png",
            )
        except Exception as exc:
            logger.debug("Skipped missed-opportunity plot %s: %s", eid, exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run(parse_args())
