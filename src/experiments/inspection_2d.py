"""2D inspection: pdSTL vs deterministic STLCG on specs R1-R3.

The map and settings come from configs/scenarios/inspection_2d.yaml.
"""

from functools import reduce

import torch

from models.rollouts import create_gaussian_belief_trajectory
from pdstl.operators import Always, Eventually, Or, Until
from planning.environment import (
    Environment,
    RectangleRegion,
    inside,
    safety_event,
)

CONFIG = "configs/scenarios/inspection_2d.yaml"
INSPECT = ("P1", "P2")


def build_environment(cfg):
    """Workspace, racks, targets P1, P2, C and goal D; the start must be safe."""
    env = Environment()
    env.set_bounds(**cfg["bounds"])
    for obstacle in cfg["obstacles"]:
        env.add_obstacle(**obstacle)
    for name, box in cfg["regions"].items():
        env.add_region(
            RectangleRegion(
                name,
                "goal" if name == "D" else "target",
                *map(float, (*box["x_range"], *box["y_range"])),
            )
        )
    # The specs start at step 1, so step 0 is checked here.
    if not satisfied(safety_event(env), [cfg["x0_mean"][:2]]):
        raise ValueError(f"start {cfg['x0_mean'][:2]} is not safe")
    return env


def build_specs(env, cfg):
    """R1-R3, each with its named parts.

    Returns:
        {"R1": (formula, parts), ...}; parts maps safety, ordering,
        inspection and deadline (those that apply) to sub-formulas, in the
        order failures are classified.
    """
    H, dwell, deadline = cfg["H"], cfg["dwell"], cfg["deadline"]
    region = {name: inside(env.region(name)) for name in cfg["regions"]}
    inspect = reduce(Or, [Always(region[n], [0, dwell]) for n in INSPECT])
    safety = Always(safety_event(env), [1, H])
    reach = Eventually(region["D"], [1, deadline])
    inspection = Eventually(inspect, [1, H - dwell])
    r1 = safety & reach
    r3 = (
        r1
        & Until(~region["C"], inspect, [1, H - dwell])
        & Until(~region["D"], region["C"], [1, deadline])
    )
    return {
        "R1": (r1, {"safety": safety, "deadline": reach}),
        "R2": (
            r1 & inspection,
            {"safety": safety, "inspection": inspection, "deadline": reach},
        ),
        "R3": (
            r3,
            {
                "safety": safety,
                "ordering": ordering(region, inspect, cfg),
                "inspection": inspection,
                "deadline": reach,
            },
        ),
    }


def ordering(region, inspect, cfg):
    """R3's order (inspect before C, C before D), for failure classes only.

    Unlike R3's strong Untils, it must not fail just because the inspection
    or C never happens: those failures belong to the inspection and
    deadline parts. R3 holds exactly when all its parts hold.

    Args:
        region: {name: inside(region)} for P1, P2, C, D.
        inspect: Dwell in P1 or P2, Always[0, dwell] P1 | Always[0, dwell] P2.
        cfg: Scenario config (H, dwell, deadline).
    """
    H, deadline = cfg["H"], cfg["deadline"]
    c, d = region["C"], region["D"]
    # Weak Until: phi W psi = (phi U psi) | Always phi.
    inspect_first = Until(~c, inspect, [1, H - cfg["dwell"]]) | Always(
        ~c, [0, H]
    )
    c_first = Until(~d, c, [1, deadline]) | Always(~d, [0, H])
    return inspect_first & c_first


def satisfied(formula, states):
    """Boolean truth at step 0 of [T, D] or [N, T, D] noise-free states."""
    states = torch.as_tensor(states, dtype=torch.float32)
    if states.ndim == 2:
        states = states.unsqueeze(0)
    beliefs = create_gaussian_belief_trajectory(
        states, torch.zeros_like(states)
    )
    return formula(beliefs)[:, 0, 0] > 0.5
