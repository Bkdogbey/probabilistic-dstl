"""The rectangular reach-avoid environment: geometry in, pdSTL specification out."""

import pytest

from pdstl.operators import Always, And, Eventually
from pdstl.predicates import InsideRectangle, OutsideRectangle
from planning.environment import Environment, RectangleRegion
from planning.environment import (
    build_reach_avoid_environment,
)

CONFIG = {
    "workspace": {"name": "workspace", "x": [0.0, 10.0], "y": [0.0, 6.0]},
    "goal": {"name": "goal", "x": [8.5, 9.5], "y": [2.5, 3.5]},
    "obstacles": [
        {"name": "upper_block", "x": [3.0, 6.0], "y": [3.5, 6.0]},
        {"name": "lower_block", "x": [3.0, 6.0], "y": [0.0, 2.5]},
    ],
}


def _config(**overrides):
    return {**{k: v for k, v in CONFIG.items()}, **overrides}


# --- Rectangle region --------------------------------------------------------------


def test_a_valid_region_keeps_its_geometry_role_and_style():
    region = RectangleRegion(
        name="goal",
        role="goal",
        xmin=1.0,
        xmax=2.0,
        ymin=3.0,
        ymax=4.5,
        style={"color": "green"},
    )

    assert (region.xmin, region.xmax, region.ymin, region.ymax) == (
        1.0,
        2.0,
        3.0,
        4.5,
    )
    assert region.x == (1.0, 2.0) and region.y == (3.0, 4.5)
    assert region.role == "goal"
    assert region.style == {"color": "green"}


@pytest.mark.parametrize(
    "bounds, axis",
    [
        (
            {"xmin": 2.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0},
            "xmin < xmax",
        ),  # reversed
        (
            {"xmin": 1.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0},
            "xmin < xmax",
        ),  # zero width
        ({"xmin": 0.0, "xmax": 1.0, "ymin": 2.0, "ymax": 1.0}, "ymin < ymax"),
        (
            {"xmin": 0.0, "xmax": 1.0, "ymin": 1.0, "ymax": 1.0},
            "ymin < ymax",
        ),  # zero height
    ],
)
def test_reversed_or_zero_width_bounds_are_rejected_by_name(bounds, axis):
    with pytest.raises(ValueError, match="block") as error:
        RectangleRegion(name="block", role="obstacle", **bounds)
    assert axis in str(error.value)


def test_an_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="role must be one of"):
        RectangleRegion(
            name="thing", role="teleporter", xmin=0, xmax=1, ymin=0, ymax=1
        )


# --- Environment container ----------------------------------------------------------


def _environment():
    return Environment()


def test_duplicate_region_names_are_rejected():
    environment = _environment()
    environment.add_region(RectangleRegion("goal", "goal", 0, 1, 0, 1))

    with pytest.raises(ValueError, match="region 'goal' already exists"):
        environment.add_region(RectangleRegion("goal", "goal", 2, 3, 2, 3))


def test_regions_are_looked_up_by_name():
    environment = _environment()
    goal = environment.add_region(RectangleRegion("goal", "goal", 0, 1, 0, 1))

    assert environment.region("goal") is goal
    with pytest.raises(ValueError, match="no region named 'nowhere'"):
        environment.region("nowhere")


def test_regions_are_looked_up_by_role_in_insertion_order():
    environment = _environment()
    environment.add_region(
        RectangleRegion("workspace", "workspace", 0, 9, 0, 9)
    )
    environment.add_region(RectangleRegion("first", "obstacle", 1, 2, 1, 2))
    environment.add_region(RectangleRegion("second", "obstacle", 3, 4, 3, 4))

    assert [r.name for r in environment.by_role("obstacle")] == [
        "first",
        "second",
    ]
    assert [r.name for r in environment.by_role("workspace")] == ["workspace"]
    assert environment.by_role("goal") == []


@pytest.mark.parametrize("horizon", [0, -1, 2.5, "10", None, True])
def test_an_invalid_horizon_is_rejected(horizon):
    environment = build_reach_avoid_environment(CONFIG)

    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        environment.get_specification(horizon)


# --- Configuration builder -----------------------------------------------------------


def test_the_builder_reads_names_bounds_and_roles_from_configuration():
    environment = build_reach_avoid_environment(CONFIG)

    assert list(environment.regions) == [
        "workspace",
        "goal",
        "upper_block",
        "lower_block",
    ]
    assert [r.name for r in environment.by_role("obstacle")] == [
        "upper_block",
        "lower_block",
    ]

    goal = environment.region("goal")
    assert (goal.xmin, goal.xmax, goal.ymin, goal.ymax) == (8.5, 9.5, 2.5, 3.5)


@pytest.mark.parametrize("missing", ["workspace", "goal"])
def test_a_missing_workspace_or_goal_is_an_error(missing):
    config = {key: value for key, value in CONFIG.items() if key != missing}

    with pytest.raises(ValueError, match=f"needs a '{missing}' block"):
        build_reach_avoid_environment(config)


@pytest.mark.parametrize("obstacles", [None, []])
def test_zero_obstacles_is_allowed(obstacles):
    environment = build_reach_avoid_environment(_config(obstacles=obstacles))

    assert environment.by_role("obstacle") == []
    specification = environment.get_specification(5)
    assert isinstance(specification.subformula1.subformula, InsideRectangle)


def test_many_obstacles_all_reach_the_formula():
    obstacles = [
        {"name": f"block_{i}", "x": [float(i), i + 0.5], "y": [0.0, 1.0]}
        for i in range(5)
    ]
    environment = build_reach_avoid_environment(_config(obstacles=obstacles))

    assert len(environment.by_role("obstacle")) == 5
    assert _leaf_names(environment.get_specification(5)) == {
        "workspace",
        "goal",
        *(f"block_{i}" for i in range(5)),
    }


def test_a_malformed_coordinate_pair_names_the_field():
    config = _config(goal={"name": "goal", "x": [1.0], "y": [0.0, 1.0]})

    with pytest.raises(
        ValueError, match="region 'goal': x must be a \\[min, max\\] pair"
    ):
        build_reach_avoid_environment(config)


def test_style_is_carried_through_untouched():
    config = _config(
        goal={
            "name": "goal",
            "x": [1.0, 2.0],
            "y": [0.0, 1.0],
            "style": {"color": "green", "hatch": "//"},
        }
    )
    environment = build_reach_avoid_environment(config)

    assert environment.region("goal").style == {
        "color": "green",
        "hatch": "//",
    }


# --- The specification ----------------------------------------------------------------


def _leaves(formula):
    """Every non-And node under a formula, flattened."""
    found, stack = [], [formula]
    while stack:
        node = stack.pop()
        if isinstance(node, And):
            stack.extend([node.subformula1, node.subformula2])
        else:
            found.append(node)
    return found


def _leaf_names(specification):
    safe = specification.subformula1.subformula
    return {leaf.name for leaf in _leaves(safe)} | {
        specification.subformula2.subformula.name
    }


def test_the_specification_is_always_safe_and_eventually_goal():
    environment = build_reach_avoid_environment(CONFIG)

    specification = environment.get_specification(12)

    assert isinstance(specification, And)
    safe, reach = specification.subformula1, specification.subformula2
    assert isinstance(safe, Always) and safe.interval == [1, 12]
    assert isinstance(reach, Eventually) and reach.interval == [0, 12]


def test_the_goal_window_is_configurable_and_validated():
    environment = build_reach_avoid_environment(CONFIG)

    specification = environment.get_specification(12, [7, 12])
    assert specification.subformula2.interval == [7, 12]

    for interval in ([7], [8, 7], [-1, 7], [7, 13], [7.0, 12]):
        with pytest.raises(ValueError, match="goal_interval"):
            environment.get_specification(12, interval)


def test_safety_conjoins_the_workspace_with_every_obstacle():
    environment = build_reach_avoid_environment(CONFIG)

    leaves = _leaves(environment.get_specification(12).subformula1.subformula)

    inside = [leaf for leaf in leaves if isinstance(leaf, InsideRectangle)]
    outside = [leaf for leaf in leaves if isinstance(leaf, OutsideRectangle)]
    assert [leaf.name for leaf in inside] == ["workspace"]
    assert {leaf.name for leaf in outside} == {"upper_block", "lower_block"}


def test_the_goal_event_is_the_configured_rectangle():
    environment = build_reach_avoid_environment(CONFIG)

    goal_event = environment.get_specification(12).subformula2.subformula

    assert isinstance(goal_event, InsideRectangle)
    assert goal_event.x_range == (8.5, 9.5)
    assert goal_event.y_range == (2.5, 3.5)


def test_moving_a_rectangle_in_configuration_moves_its_event():
    moved = _config(
        obstacles=[{"name": "block", "x": [1.25, 2.75], "y": [3.5, 4.5]}]
    )

    (event,) = [
        leaf
        for leaf in _leaves(
            build_reach_avoid_environment(moved)
            .get_specification(5)
            .subformula1.subformula
        )
        if isinstance(leaf, OutsideRectangle)
    ]

    assert event.x_range == (1.25, 2.75)
    assert event.y_range == (3.5, 4.5)


def test_the_environment_computes_no_probabilities_and_draws_nothing():
    """Sampling, robustness and plotting belong to other layers."""
    environment = build_reach_avoid_environment(CONFIG)

    for attribute in (
        "draw_on_ax",
        "metadata",
        "sample",
        "robustness",
        "probability",
    ):
        assert not hasattr(environment, attribute), (
            f"Environment grew {attribute!r}"
        )
    assert set(vars(environment)) == {"regions"}


def test_planner_facing_usage_is_two_calls():
    environment = build_reach_avoid_environment(CONFIG)
    specification = environment.get_specification(20)

    assert specification is not None
