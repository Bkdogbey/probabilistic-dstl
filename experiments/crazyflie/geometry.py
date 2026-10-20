"""Arena geometry and offline clearance checks; no optimizer dependencies."""

import minsnap
import numpy as np


def _box_distance(xy, box):
    """Signed distance from points [..., 2] to a box: > 0 outside."""
    lo, hi = np.array([box.x[0], box.y[0]]), np.array([box.x[1], box.y[1]])
    d = np.maximum(lo - xy, xy - hi)
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    return np.where(d.max(-1) > 0, outside, d.max(-1))


class Task:
    """Stay in the workspace, avoid obstacles, reach the goal.

    Obstacles grow and the workspace shrinks by the airframe radius, so
    every check is on the drone's centre. The goal must be reached at some
    step 1..H (the figure-8 passes its top tip and returns to the start).
    """

    def __init__(self, cfg):
        r = cfg.airframe_radius
        self.workspace = cfg.workspace.grown(-r)
        self.obstacles = [box.grown(r) for box in cfg.obstacles]
        self.goal = cfg.goal
        self.steps = cfg.steps
        self._env = None

    @property
    def env(self):
        # Geometry and replay do not import the torch planning stack.
        if self._env is None:
            from planning.environment import Environment

            env = Environment()
            env.set_bounds(self.workspace.x, self.workspace.y)
            for box in self.obstacles:
                env.add_obstacle(box.x, box.y, name=box.name)
            env.set_goal(self.goal.x, self.goal.y, name="goal")
            self._env = env
        return self._env

    @property
    def spec(self):
        return self.env.get_specification(self.steps)

    def clearance(self, xy):
        """Airframe clearance [m] of positions [..., 2]; < 0 is a violation."""
        xy = np.asarray(xy, dtype=float)
        walls = -_box_distance(xy, self.workspace)
        return np.min(
            [walls, *(_box_distance(xy, box) for box in self.obstacles)],
            axis=0,
        )

    def requirements(self, path):
        """Each requirement on a path [N, 2] from its start: met or not."""
        met = {
            "workspace": bool(
                (-_box_distance(path, self.workspace) >= 0).all()
            )
        }
        for box in self.obstacles:
            met[box.name] = bool((_box_distance(path, box) >= 0).all())
        met["goal"] = bool((_box_distance(path[1:], self.goal) <= 0).any())
        return met


def check_feasible(cfg, task, plan, mean=None):
    """Execution checks of a plan's uploaded rows; problems, or [].

    Sampled at minsnap.DENSE_HZ plus every piece boundary: these are sampled
    checks, not a proof of clearance between samples. `mean` [H+1, 2] is
    the predicted mean on the planning grid.
    """
    rows = np.asarray(plan["coefficients"], dtype=float)
    problems = minsnap.problems(rows, plan["waypoints"])
    if len(rows) > minsnap.MAX_PIECES:
        problems.append(
            f"{len(rows)} pieces, trajectory memory holds {minsnap.MAX_PIECES}"
        )
    t = minsnap.dense_times(rows)
    speed = np.linalg.norm(minsnap.evaluate(rows, t, 1)[:, :2], axis=1).max()
    if speed > cfg.v_max:
        problems.append(f"speed {speed:.3f} above v_max = {cfg.v_max} m/s")
    accel = np.linalg.norm(minsnap.evaluate(rows, t, 2)[:, :2], axis=1).max()
    if accel > cfg.a_max:
        problems.append(
            f"acceleration {accel:.3f} above a_max = {cfg.a_max} m/s^2"
        )
    curve = minsnap.evaluate(rows, t)[:, :2]
    for label, path in (("curve", curve), ("predicted mean", mean)):
        if path is None:
            continue
        clearance = task.clearance(path).min()
        if clearance < cfg.clearance_margin:
            problems.append(
                f"{label} clearance {clearance:.3f} m "
                f"below {cfg.clearance_margin} m"
            )
    if np.linalg.norm(curve[0] - np.asarray(cfg.start)) > minsnap.WAYPOINT_TOL:
        problems.append(f"curve does not start at {list(cfg.start)}")
    duration = minsnap.total_duration(rows)
    if abs(duration - cfg.duration) > 1e-6:
        problems.append(
            f"curve lasts {duration} s, mission needs {cfg.duration} s"
        )
    return problems
