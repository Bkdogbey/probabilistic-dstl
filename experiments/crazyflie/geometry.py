"""Arena geometry and offline clearance checks; no optimizer dependencies."""

import numpy as np


def _box_distance(xy, box):
    """Signed distance from points [..., 2] to a box: > 0 outside."""
    lo, hi = np.array([box.x[0], box.y[0]]), np.array([box.x[1], box.y[1]])
    d = np.maximum(lo - xy, xy - hi)
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    return np.where(d.max(-1) > 0, outside, d.max(-1))


class Task:
    """Stay in the workspace, avoid obstacles, end inside the goal.

    Obstacles grow and the workspace shrinks by the airframe radius, so
    every check is on the drone's centre. The goal is required only at the
    final step H, so starting inside it never counts as completion.
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
            env.set_goal(
                self.goal.x,
                self.goal.y,
                interval=[self.steps] * 2,
                name="goal",
            )
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

    def requirements(self, path, final):
        """Each requirement on a path [N, 2] ending at `final` [2]: met or not."""
        met = {
            "workspace": bool(
                (-_box_distance(path, self.workspace) >= 0).all()
            )
        }
        for box in self.obstacles:
            met[box.name] = bool((_box_distance(path, box) >= 0).all())
        met["goal"] = bool(_box_distance(np.asarray(final), self.goal) <= 0)
        return met


def nominal_path(cfg, start, velocities):
    """Positions at the streaming rate if the drone tracked v exactly."""
    v = np.repeat(np.asarray(velocities), cfg.repeats, axis=0)
    steps = np.cumsum(v / cfg.stream_hz, axis=0)
    return np.vstack([start, np.asarray(start) + steps])


def check_feasible(cfg, task, start, velocities, mean=None, mission=True):
    """Physical feasibility shared by every condition; problems, or []."""
    v = np.asarray(velocities)
    problems = []
    if np.abs(v).max() > cfg.u_max + 1e-4:
        problems.append(f"command above u_max = {cfg.u_max} m/s")
    accel = np.abs(np.diff(np.vstack([[0.0, 0.0], v]), axis=0)).max() / cfg.dt
    if accel > cfg.a_max + 1e-4:
        problems.append(
            f"acceleration {accel:.2f} above a_max = {cfg.a_max} m/s^2"
        )
    for label, path in (
        ("nominal path", nominal_path(cfg, start, v)),
        ("predicted mean", mean),
    ):
        if path is not None and task.clearance(path).min() < 0:
            problems.append(f"{label} leaves the free space")
    if mission and len(v) != cfg.steps:
        problems.append(f"{len(v)} steps, mission needs {cfg.steps}")
    return problems
