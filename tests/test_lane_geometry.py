import numpy as np

from src.features.lane import point_to_segment_distance


def test_point_to_segment_distance_for_perpendicular_projection():
    dist = point_to_segment_distance(px=5.0, py=2.0, ax=0.0, ay=0.0, bx=10.0, by=0.0)

    assert np.isclose(dist, 2.0)


def test_point_to_segment_distance_clamps_before_segment():
    dist = point_to_segment_distance(px=-3.0, py=4.0, ax=0.0, ay=0.0, bx=10.0, by=0.0)

    assert np.isclose(dist, 5.0)


def test_point_to_segment_distance_handles_zero_length_segment():
    dist = point_to_segment_distance(px=3.0, py=4.0, ax=0.0, ay=0.0, bx=0.0, by=0.0)

    assert np.isclose(dist, 5.0)
