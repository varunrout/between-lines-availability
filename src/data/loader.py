"""
Phase 1 – Data loading from StatsBomb open data.

Loads events, 360 freeze frames, lineups and match metadata for the
configured competitions.  All data is returned as plain pandas DataFrames.

Usage
-----
    from src.data.loader import load_competition, load_all_competitions

    events, frames, lineups, matches = load_competition(
        competition_id=55, season_id=43
    )
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

from src.config import COMPETITIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_competition(
    competition_id: int,
    season_id: int,
    include_lineups: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all data for a single competition season.

    Parameters
    ----------
    competition_id:
        StatsBomb competition identifier.
    season_id:
        StatsBomb season identifier.
    include_lineups:
        Whether to also load lineups (adds extra API calls).

    Returns
    -------
    events_df, frames_df, lineups_df, matches_df
        * events_df   – one row per on-ball event
        * frames_df   – one row per 360 freeze-frame player entry
        * lineups_df  – one row per player per match
        * matches_df  – one row per match
    """
    logger.info("Loading matches for competition=%d season=%d", competition_id, season_id)
    matches_df = sb.matches(competition_id=competition_id, season_id=season_id)
    matches_df["competition_id"] = competition_id
    matches_df["season_id"] = season_id

    all_events: list[pd.DataFrame] = []
    all_frames: list[pd.DataFrame] = []
    all_lineups: list[pd.DataFrame] = []

    for match_id in tqdm(matches_df["match_id"], desc=f"comp={competition_id} season={season_id}"):
        # ----- events -----
        try:
            ev = sb.events(match_id=match_id, split=False, flatten_attrs=True)
            ev["match_id"] = match_id
            all_events.append(ev)
        except Exception as exc:
            logger.warning("Could not load events for match %s: %s", match_id, exc)
            continue

        # ----- 360 frames -----
        try:
            fr = _load_frames(match_id)
            if fr is not None:
                all_frames.append(fr)
        except Exception as exc:
            logger.warning("Could not load frames for match %s: %s", match_id, exc)

        # ----- lineups -----
        if include_lineups:
            try:
                lu = sb.lineups(match_id=match_id)
                for team_name, team_df in lu.items():
                    team_df["team_name"] = team_name
                    team_df["match_id"] = match_id
                    all_lineups.append(team_df)
            except Exception as exc:
                logger.warning("Could not load lineups for match %s: %s", match_id, exc)

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    frames_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    lineups_df = pd.concat(all_lineups, ignore_index=True) if all_lineups else pd.DataFrame()

    logger.info(
        "Loaded %d events, %d frame rows, %d lineup rows across %d matches",
        len(events_df),
        len(frames_df),
        len(lineups_df),
        len(matches_df),
    )
    return events_df, frames_df, lineups_df, matches_df


def load_all_competitions(
    include_lineups: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data for all competitions defined in config.COMPETITIONS.

    Returns concatenated events_df, frames_df, lineups_df, matches_df.
    """
    all_events, all_frames, all_lineups, all_matches = [], [], [], []

    for comp in COMPETITIONS:
        try:
            ev, fr, lu, ma = load_competition(
                competition_id=comp["competition_id"],
                season_id=comp["season_id"],
                include_lineups=include_lineups,
            )
            ev["competition_name"] = comp["name"]
            fr["competition_name"] = comp["name"] if not fr.empty else None
            lu["competition_name"] = comp["name"] if not lu.empty else None
            ma["competition_name"] = comp["name"]

            all_events.append(ev)
            all_frames.append(fr)
            all_lineups.append(lu)
            all_matches.append(ma)
        except Exception as exc:
            logger.error("Failed to load competition %s: %s", comp["name"], exc)

    return (
        pd.concat(all_events, ignore_index=True),
        pd.concat(all_frames, ignore_index=True),
        pd.concat(all_lineups, ignore_index=True),
        pd.concat(all_matches, ignore_index=True),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_frames(match_id: int) -> pd.DataFrame | None:
    """Load and parse 360 freeze-frame data for a single match.

    Returns a tidy DataFrame with one row per player per event, plus
    separate ``visible_area`` and ``event_id`` columns.

    The raw StatsBomb 360 JSON has this shape per event entry::

        {
          "id": "<event_uuid>",
          "freeze_frame": [
            {"location": [x, y], "player": {...}, "position": {...}, "teammate": bool}
          ],
          "visible_area": [x1, y1, x2, y2, ...]  # flat list
        }
    """
    raw = sb.frames(match_id=match_id, fmt="dataframe")

    if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
        return None

    rows: list[dict] = []
    for _, event_row in raw.iterrows():
        event_id = event_row["id"]
        visible_area_raw = event_row.get("visible_area")
        freeze_frame = event_row.get("freeze_frame", [])

        # Parse visible_area – stored as flat [x1,y1,x2,y2,...] list
        visible_area_coords = _parse_visible_area(visible_area_raw)

        for player in freeze_frame:
            loc = player.get("location", [None, None])
            position = player.get("position") or {}
            player_info = player.get("player") or {}
            rows.append(
                {
                    "event_id": event_id,
                    "match_id": match_id,
                    "player_id": player_info.get("id"),
                    "player_name": player_info.get("name"),
                    "position_id": position.get("id"),
                    "position_name": position.get("name"),
                    "teammate": player.get("teammate", False),
                    "frame_x": loc[0] if len(loc) > 0 else None,
                    "frame_y": loc[1] if len(loc) > 1 else None,
                    "visible_area": visible_area_coords,
                }
            )

    return pd.DataFrame(rows) if rows else None


def _parse_visible_area(raw) -> list[tuple[float, float]] | None:
    """Convert raw visible_area data to a list of (x, y) tuples.

    Handles both flat lists [x1,y1,x2,y2,...] and nested [[x1,y1],...] formats.
    """
    if raw is None:
        return None
    if isinstance(raw, list) and len(raw) > 0:
        if isinstance(raw[0], (list, tuple)):
            # Already nested
            return [(float(pt[0]), float(pt[1])) for pt in raw]
        else:
            # Flat list – pair up
            return [
                (float(raw[i]), float(raw[i + 1]))
                for i in range(0, len(raw) - 1, 2)
            ]
    return None
