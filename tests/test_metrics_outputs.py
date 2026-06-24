import pandas as pd

from src.metrics.outputs import compute_player_metrics, compute_team_metrics, merge_event_scores


def test_merge_event_scores_handles_empty_event_scores_and_lines():
    buildup = pd.DataFrame(
        [
            {
                "id": "event-1",
                "team": "Spain",
                "competition_name": "UEFA Euro 2024",
                "x": 45.0,
                "y": 40.0,
            }
        ]
    )

    merged = merge_event_scores(buildup, pd.DataFrame(), pd.DataFrame())

    assert len(merged) == 1
    assert merged.loc[0, "findable_option_available"] == 0
    assert merged.loc[0, "n_findable_candidates"] == 0
    assert merged.loc[0, "n_between_lines_candidates"] == 0


def test_compute_team_metrics_produces_percentage_rates():
    merged = pd.DataFrame(
        [
            {
                "team": "Spain",
                "competition_name": "UEFA Euro 2024",
                "findable_option_available": 1,
                "n_findable_candidates": 2,
                "n_between_lines_candidates": 3,
                "central_findable_count": 1,
                "half_space_findable_count": 1,
                "line_gap_depth": 14.0,
            },
            {
                "team": "Spain",
                "competition_name": "UEFA Euro 2024",
                "findable_option_available": 0,
                "n_findable_candidates": 0,
                "n_between_lines_candidates": 1,
                "central_findable_count": 0,
                "half_space_findable_count": 0,
                "line_gap_depth": 10.0,
            },
        ]
    )

    metrics = compute_team_metrics(merged)

    assert metrics.loc[0, "team"] == "Spain"
    assert metrics.loc[0, "total_buildup_events"] == 2
    assert metrics.loc[0, "between_lines_availability_rate"] == 50.0
    assert metrics.loc[0, "avg_line_gap"] == 12.0


def test_compute_player_metrics_handles_empty_receiver_table():
    metrics = compute_player_metrics(pd.DataFrame(), pd.DataFrame())

    assert metrics.empty
    assert "receiver_player_name" in metrics.columns
