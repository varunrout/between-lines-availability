import pandas as pd

from src.features.findability import EVENT_SCORE_COLUMNS, compute_findability, score_candidate


def _candidate(**overrides):
    row = {
        "between_lines": True,
        "ahead_of_ball": True,
        "in_visible_area": True,
        "centrality_zone": "central",
        "pass_distance": 18.0,
        "nearest_defender_dist": 4.0,
        "lane_blocked": 0,
    }
    row.update(overrides)
    return pd.Series(row)


def test_score_candidate_requires_core_between_lines_conditions():
    assert score_candidate(_candidate()) == 3
    assert score_candidate(_candidate(between_lines=False)) == 0
    assert score_candidate(_candidate(ahead_of_ball=False)) == 0
    assert score_candidate(_candidate(in_visible_area=False)) == 0
    assert score_candidate(_candidate(centrality_zone="wide")) == 0


def test_score_candidate_degrades_for_risk_factors():
    assert score_candidate(_candidate(pass_distance=35.0)) == 2
    assert score_candidate(_candidate(nearest_defender_dist=1.0)) == 2
    assert score_candidate(_candidate(lane_blocked=1)) == 2


def test_compute_findability_returns_stable_schema_when_empty():
    event_scores, candidates_scored = compute_findability(pd.DataFrame())

    assert list(event_scores.columns) == EVENT_SCORE_COLUMNS
    assert candidates_scored.empty
    assert "findability_score" in candidates_scored.columns


def test_compute_findability_aggregates_event_level_availability():
    candidates = pd.DataFrame(
        [
            {
                "event_id": "event-1",
                "receiver_player_name": "Player A",
                "receiver_x": 55.0,
                "receiver_y": 40.0,
                "between_lines": True,
                "ahead_of_ball": True,
                "in_visible_area": True,
                "centrality_zone": "central",
                "pass_distance": 20.0,
                "nearest_defender_dist": 4.0,
                "lane_blocked": 0,
            },
            {
                "event_id": "event-1",
                "receiver_player_name": "Player B",
                "receiver_x": 58.0,
                "receiver_y": 70.0,
                "between_lines": True,
                "ahead_of_ball": True,
                "in_visible_area": True,
                "centrality_zone": "wide",
                "pass_distance": 12.0,
                "nearest_defender_dist": 6.0,
                "lane_blocked": 0,
            },
        ]
    )

    event_scores, candidates_scored = compute_findability(candidates)

    assert candidates_scored["findability_score"].tolist() == [3, 0]
    assert event_scores.loc[0, "findable_option_available"] == 1
    assert event_scores.loc[0, "n_findable_candidates"] == 1
    assert event_scores.loc[0, "central_findable_count"] == 1
