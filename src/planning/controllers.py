"""Receding-horizon execution: repeatedly solve one window and apply its first controls.

There is exactly one MPC implementation, and every window goes through `Planner.solve()`.
The controller owns the outer loop and nothing else -- no scenario knowledge, no geometry,
no estimator.
"""

from dataclasses import dataclass

import torch

from planning.environment import DeadlineExpired
from planning.planner import PlanResult

CONTROLLER_ALIASES = {"mpc": "receding_horizon"}


def normalize_controller_type(name):
    """`mpc` is a backward-compatible alias; both names route to the same implementation."""
    return CONTROLLER_ALIASES.get(name, name)


def shift_controls(controls, applied_steps=1):
    """Drop the applied controls and hold the last one, for the next window's warm start."""
    if applied_steps <= 0:
        return controls
    remaining = controls[applied_steps:]
    padding = controls[-1:].repeat(applied_steps, 1)
    return torch.cat([remaining, padding], dim=0)


@dataclass
class RecedingHorizonResult:
    """Execution trace. Per-window detail stays inside `plans[k]`, never copied out."""

    states: torch.Tensor                 # [steps + 1, D] realised states
    beliefs: list                        # [(mean, covariance)] per state
    applied_controls: torch.Tensor       # [steps, m]
    plans: list[PlanResult]
    stopped_reason: str

    @property
    def hard_lowers(self):
        """Per-window exact hard lower score, derived rather than stored."""
        return [plan.hard_lower for plan in self.plans]


class RecedingHorizonController:
    """Plan a window, apply its first controls, observe, replan."""

    def __init__(self, planner, dynamics, scenario, config=None):
        self.planner = planner
        self.dynamics = dynamics
        self.scenario = scenario
        self.config = dict(config or {})

    def run(self, initial_state, initial_belief, *, max_steps, apply_steps=1,
            initial_controls=None, on_step=None):
        warm_start_enabled = self.config.get("warm_start", True)
        state, belief = initial_state, initial_belief

        states, beliefs, applied, plans = [state], [belief], [], []
        # The first window has no previous solution to shift, so it takes the caller's guess.
        warm_start, stopped_reason = initial_controls, "max_steps"

        if hasattr(self.scenario, "reset_progress"):
            self.scenario.reset_progress()

        for step in range(max_steps):
            window = self.scenario.window(step=step, belief=belief)
            try:
                specification = window.specification(
                    self.planner.horizon, context={"step": step}
                )
            except DeadlineExpired:
                stopped_reason = "deadline_expired"
                break
            plan = self.planner.solve(
                belief[0], belief[1],
                initial_controls=warm_start,
                specification=specification,
                extra_loss=window.extra_loss,
            )
            plans.append(plan)

            done = False
            for control in plan.controls[:apply_steps]:
                state, belief = self.dynamics.execute_step(state, belief, control)
                states.append(state)
                beliefs.append(belief)
                applied.append(control)
                if self.scenario.is_complete(state, belief):
                    stopped_reason, done = "scenario_complete", True
                    break

            if on_step is not None:
                on_step(step, state, belief, plan)
            if done:
                break

            warm_start = shift_controls(plan.controls, apply_steps) if warm_start_enabled else None

        return RecedingHorizonResult(
            states=torch.stack(states),
            beliefs=beliefs,
            applied_controls=(
                torch.stack(applied)
                if applied
                else torch.empty(0, self.planner.control_dim, device=self.planner.device)
            ),
            plans=plans,
            stopped_reason=stopped_reason,
        )
