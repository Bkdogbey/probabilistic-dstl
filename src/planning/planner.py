"""One optimizer and one receding-horizon loop over caller-supplied belief rollouts."""

import logging
from dataclasses import dataclass

import torch

from models.rollouts import BeliefRollout
from utils import load_config

logger = logging.getLogger(__name__)


@dataclass
class PlanResult:
    controls: torch.Tensor  # [H, control_dim], physical controls
    rollout: BeliefRollout
    smooth_lower: float
    hard_interval: tuple[float, float]
    loss_history: list[float]


@dataclass
class MPCResult:
    states: list
    applied_controls: torch.Tensor  # [steps, control_dim]
    window_plans: list[PlanResult]
    stopped_reason: str


class Planner:
    """Optimize controls without knowing how the rollout represents uncertainty.

    rollout(v) returns a BeliefRollout; dynamics supplies bounds, device, dtype
    and control dimension. Only the smooth lower score is optimized.
    """

    def __init__(self, dynamics, horizon, config=None):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self.dyn, self.horizon = dynamics, horizon
        self.device = dynamics.device
        defaults = load_config("configs/planning.yaml")
        config = config or {}
        unknown = set(config) - set(defaults)
        if unknown:
            raise ValueError(f"unknown planner settings: {sorted(unknown)}")
        self.cfg = {**defaults, **config}
        self.cfg["smoothing"] = {**defaults["smoothing"], **config.get("smoothing", {})}
        unknown_smoothing = set(self.cfg["smoothing"]) - set(defaults["smoothing"])
        if unknown_smoothing:
            raise ValueError(f"unknown smoothing settings: {sorted(unknown_smoothing)}")
        if self.cfg["max_iters"] < 1:
            raise ValueError("max_iters must be positive")
        if any(self.cfg["smoothing"][key] <= 0 for key in ("beta_start", "beta_end")):
            raise ValueError("smoothing beta must be positive")

    @property
    def control_dim(self):
        return self.dyn.B.shape[1]

    def _control_parameters(self, controls):
        controls = torch.as_tensor(controls, device=self.device, dtype=self.dyn.B.dtype)
        if controls.shape != (self.horizon, self.control_dim):
            raise ValueError("controls must have shape [horizon, control_dim]")
        if not torch.isfinite(controls).all() or (controls.abs() > self.dyn.u_max).any():
            raise ValueError("controls must be finite and within the control bounds")
        # A finite inverse is necessary at exact saturation.
        margin = torch.finfo(controls.dtype).eps
        return torch.atanh((controls / self.dyn.u_max).clamp(-1 + margin, 1 - margin))

    def _init_controls(self, init_guess):
        values = (torch.zeros(self.horizon, self.control_dim, device=self.device,
                              dtype=self.dyn.B.dtype)
                  if init_guess is None else self._control_parameters(init_guess))
        return torch.nn.Parameter(values.detach().clone())

    def _control_cost(self, controls):
        smoothness = (controls[1:] - controls[:-1]).square().sum() + controls[0].square().sum()
        return self.cfg["w_u"] * controls.square().sum() + self.cfg["w_du"] * smoothness

    def _objective(self, controls, smooth_lower):
        return -self.cfg["w_phi"] * smooth_lower + self._control_cost(controls)

    def _beta(self, iteration):
        smoothing = self.cfg["smoothing"]
        start, end = smoothing["beta_start"], smoothing["beta_end"]
        return start * (end / start) ** (iteration / max(self.cfg["max_iters"] - 1, 1))

    def _evaluate(self, rollout, parameters, spec, beta, history):
        with torch.no_grad():
            prediction = rollout(parameters)
            smooth = spec.smooth_lower(prediction.belief_trajectory, beta).item()
            interval = spec.probability_interval(prediction.belief_trajectory)
            return PlanResult(self.dyn.bound_control(parameters).detach().clone(),
                              prediction, smooth, tuple(interval.tolist()), list(history))

    def evaluate_controls(self, rollout, controls, *, spec):
        """Replay physical controls using final smoothing and exact reporting semantics."""
        return self._evaluate(rollout, self._control_parameters(controls), spec,
                              self.cfg["smoothing"]["beta_end"], [])

    def optimize_window(self, rollout, *, spec, init_guess=None, verbose=False,
                        on_iteration=None):
        """Return the final iterate, never a checkpoint chosen by another objective.

        An optional observer receives (iteration, PlanResult) after each update.
        Observer scores use that iteration's beta; the returned score uses beta_end.
        """
        parameters = self._init_controls(init_guess)
        optimizer = torch.optim.Adam([parameters], lr=self.cfg["lr"])
        history, patience = [], 0
        for iteration in range(self.cfg["max_iters"]):
            beta = self._beta(iteration)
            optimizer.zero_grad()
            prediction = rollout(parameters)
            lower = spec.smooth_lower(prediction.belief_trajectory, beta)
            loss = self._objective(self.dyn.bound_control(parameters), lower)
            if not torch.isfinite(loss):
                raise ValueError("planning loss is not finite")
            loss.backward()
            if parameters.grad is None or not torch.isfinite(parameters.grad).all():
                raise ValueError("rollout must supply finite gradients to controls")
            optimizer.step()
            history.append(loss.item())
            threshold = self.cfg["alpha"]
            if on_iteration is not None or threshold is not None:
                observed = self._evaluate(rollout, parameters, spec, beta, history)
                if on_iteration is not None:
                    on_iteration(iteration, observed)
                if threshold is not None:
                    patience = patience + 1 if observed.hard_interval[0] >= threshold else 0
                    if patience >= self.cfg["converge_patience"]:
                        break
            if verbose and iteration % 50 == 0:
                logger.info("Iteration %d: loss %.6f, smooth lower %.6f",
                            iteration, loss.item(), lower.item())
            if (iteration >= max(1, self.cfg["min_iters"])
                    and beta == self._beta(iteration - 1)
                    and abs(history[-1] - history[-2]) < self.cfg["loss_tol"]):
                break
        return self._evaluate(rollout, parameters, spec,
                              self.cfg["smoothing"]["beta_end"], history)

    @staticmethod
    def shift_controls(controls):
        """Discard the executed control and hold the last control at the horizon tail."""
        return torch.cat((controls[1:], controls[-1:])).detach().clone()

    def run_receding_horizon(self, state, *, make_rollout, execute, make_spec,
                            is_done, max_steps, init_guess=None, verbose=False,
                            on_step=None):
        """Callbacks use (state, step); execute additionally receives physical u_0.

        execute returns a new state rather than mutating recorded states.
        is_done is pure. on_step(step, updated_state, plan) only observes.
        """
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        states, applied, plans = [state], [], []
        guess = init_guess
        done = bool(is_done(state, 0))
        for step in range(max_steps):
            if done:
                break
            plan = self.optimize_window(make_rollout(state, step),
                                        spec=make_spec(state, step), init_guess=guess,
                                        verbose=verbose)
            control = plan.controls[0].clone()
            state = execute(state, control, step)
            states.append(state)
            applied.append(control)
            plans.append(plan)
            guess = self.shift_controls(plan.controls)
            done = bool(is_done(state, step + 1))
            if on_step is not None:
                on_step(step, state, plan)
        controls = (torch.stack(applied) if applied else
                    torch.empty(0, self.control_dim, device=self.device, dtype=self.dyn.B.dtype))
        return MPCResult(states, controls, plans, "goal_reached" if done else "max_steps")
