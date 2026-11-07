import numpy as np
import pytest

from backend.routing import RouteError, nearest_next


def crosses_wall_only_at_gap(points, gap_z):
    """The first coordinate crosses the wall at x = 1 only inside the gap."""
    points = np.array(points)
    crossed = np.flatnonzero(np.diff(np.sign(points[:, 0] - 1.0)))
    return len(crossed) >= 1 and all(gap_z[0] <= points[i, 2] <= gap_z[1] or gap_z[0] <= points[i + 1, 2] <= gap_z[1] for i in crossed)


def test_fine_grid_goes_through_the_gap(scene):
    route = scene.grid("fine").route([0.2, 1.0, 0.2], scene.item("P-001"))
    assert route.length > 1.5
    assert crosses_wall_only_at_gap(route.points, (1.4, 1.8))


def test_cell_grid_goes_through_the_gap(scene):
    route = scene.grid("cells").route([0.2, 1.0, 0.2], scene.item("P-001"))
    assert crosses_wall_only_at_gap(route.points, (1.5, 1.9))
    assert route.end_cell is not None


def test_cell_numbers_round_trip_and_match_the_top_view(scene):
    grid = scene.grid("cells")
    assert grid.index_of(1) == (0, 9)
    assert grid.index_of(100) == (9, 0)
    assert all(grid.number_of(grid.index_of(n)) == n for n in range(1, 101))


def test_blocked_and_missing_cells_are_refused(scene):
    grid = scene.grid("cells")
    blocked = grid.number_of((5, 0))
    with pytest.raises(RouteError, match="blocked"):
        grid.route_to_cell([0.2, 1, 0.2], blocked)
    with pytest.raises(RouteError, match="does not exist"):
        grid.route_to_cell([0.2, 1, 0.2], 101)


def test_start_on_a_blocked_spot_moves_to_free_floor(scene):
    route = scene.grid("cells").route([1.05, 1.0, 0.1], scene.item("CV-001"))
    assert route.steps >= 1


def test_no_route_is_reported(scene):
    grid = scene.grid("cells")
    grid.sizes[5, 8] = 0
    with pytest.raises(RouteError, match="no walkable route"):
        grid.route([0.2, 1.0, 0.2], scene.item("P-001"))


def test_unknown_grid_and_item_are_refused(scene):
    with pytest.raises(RouteError):
        scene.grid("nope")
    with pytest.raises(RouteError):
        scene.item("X-999")


def test_nearest_next_visits_the_closest_first(scene):
    legs = nearest_next(scene.grid("fine"), [0.2, 1.0, 0.2], scene.assets())
    assert [item["tag"] for item, _ in legs] == ["CV-001", "P-001"]
