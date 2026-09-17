"""Planning world: static rectangular geometry and the pdSTL formula it implies.

Geometry only. This module computes no probabilities, owns no dynamics, runs no optimization
and holds no results: it names events, and the belief evaluates them.

The scenario file supplies every coordinate. Nothing here hard-codes a workspace, a goal or a
block, so a rectangle can be added, moved or resized by editing YAML alone.
"""

from functools import reduce

from pdstl.operators import Always, And, Eventually
from pdstl.predicates import InsideRectangle, OutsideRectangle


def _rectangle(spec, label):
    """(name, x_range, y_range) from a config block."""
    try:
        return spec.get("name", label), spec["x_range"], spec["y_range"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{label} needs x_range and y_range, got {spec!r}") from error


class DeadlineExpired(Exception):
    """Raised when an absolute temporal obligation can no longer be met."""


class Environment:
    """Workspace bounds, one goal rectangle and any number of static obstacle rectangles."""

    def __init__(self, device="cpu"):
        self.obstacles = []
        self.goal = None
        self.bounds = None
        self.goal_deadline = None  # absolute mission deadline; None means "the window horizon"
        self.device = device

    # --- Construction ----------------------------------------------------------

    @classmethod
    def from_config(cls, config, device="cpu"):
        """Build from the `environment:` block of a scenario file."""
        env = cls(device=device)
        env.goal_deadline = config.get("goal_deadline")
        if "bounds" in config:
            env.set_bounds(**config["bounds"])
        if config.get("goal") is not None:
            name, x_range, y_range = _rectangle(config["goal"], "goal")
            env.set_goal(x_range, y_range, name=name)
        for index, obstacle in enumerate(config.get("obstacles") or []):
            name, x_range, y_range = _rectangle(obstacle, f"obstacle_{index}")
            env.add_obstacle(x_range, y_range, name=name)
        return env

    def add_obstacle(self, x_range, y_range, name=None):
        self.obstacles.append(
            {"x": x_range, "y": y_range, "name": name or f"obstacle_{len(self.obstacles)}"}
        )

    def set_goal(self, x_range, y_range, name="goal"):
        self.goal = {"x": x_range, "y": y_range, "name": name}

    def set_bounds(self, x_range, y_range):
        """Workspace the trajectory must always stay inside."""
        self.bounds = {"x": x_range, "y": y_range}

    # --- Specification ---------------------------------------------------------

    def predicates(self, context=None):
        """Named geometric events. `context` is unused for static geometry."""
        return {
            "obstacles": [
                OutsideRectangle(o["x"], o["y"], name=f"outside_{o['name']}")
                for o in self.obstacles
            ],
            "workspace": (
                InsideRectangle(self.bounds["x"], self.bounds["y"], name="inside_workspace")
                if self.bounds
                else None
            ),
            "goal": (
                InsideRectangle(self.goal["x"], self.goal["y"], name=f"inside_{self.goal['name']}")
                if self.goal
                else None
            ),
        }

    def goal_window(self, horizon, step=0):
        """Remaining goal interval at execution step k: [max(0, a - k), b - k].

        With no configured `goal_deadline` the obligation is window-relative and every replan
        gets [1, horizon]. With one, the deadline is absolute and counts down, so receding-horizon
        execution cannot quietly restart the clock each time it replans.
        """
        if self.goal_deadline is None:
            return [1, horizon]
        start, deadline = 1, int(self.goal_deadline)
        remaining = deadline - step
        if remaining < 0:
            raise DeadlineExpired(
                f"goal deadline {deadline} passed at execution step {step}"
            )
        return [min(max(0, start - step), horizon), min(remaining, horizon)]

    def specification(self, horizon, context=None):
        """Always[1,H](every obstacle avoided and inside the workspace) and Eventually(goal).

        Safety and the workspace share one Always so the conjunction is taken per step, before
        the temporal reduction. Always/Eventually keep the project's pointwise min/max StoRI
        semantics; the result is not a joint trajectory-satisfaction probability.
        """
        step = int((context or {}).get("step", 0))
        events = self.predicates(context)
        invariants = list(events["obstacles"])
        if events["workspace"] is not None:
            invariants.append(events["workspace"])

        clauses = []
        if invariants:
            clauses.append(Always(reduce(And, invariants), interval=[1, horizon]))
        if events["goal"] is not None:
            clauses.append(Eventually(events["goal"], interval=self.goal_window(horizon, step)))
        if not clauses:
            raise ValueError("environment defines no obstacles, workspace or goal")
        return reduce(And, clauses)

    def window(self, step=0, belief=None):
        """The environment to plan against at this execution step. Static geometry: itself."""
        return self

    def extra_loss(self, mean_trace, config):
        """Scenario-supplied shaping. The canonical objective adds nothing."""
        return None

    # --- Execution / drawing ---------------------------------------------------

    def is_complete(self, state, belief, context=None):
        """Static reach-avoid has no early-exit condition; the horizon is the whole plan."""
        return False

    def draw_on_ax(self, ax, context=None, **kwargs):
        from visualization.planning import draw_env_on_ax  # keeps matplotlib out of import

        draw_env_on_ax(ax, self, **kwargs)
