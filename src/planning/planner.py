"""Gradient ascent on the smooth pdSTL lower bound, and a receding-horizon loop."""

import logging
from dataclasses import dataclass, field
from time import perf_counter

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
    final_loss: float
    smoothing_beta: float
    planning_time: float = 0.0
    control_cost: float = 0.0
    alpha: float | None = None
    threshold_met: bool | None = None
    selected_iteration: int = -1
    hard_lower_history: list[float] = field(default_factory=list)
    initial_hard_lower: float | None = None
    initial_smooth_lower: float | None = None
    monte_carlo: tuple[float, float, float] | None = None

    def alpha_status(self):
        """Whether the lower robustness meets alpha, in words."""
        if self.alpha is None:
            return "no α requested"
        relation = "meets" if self.threshold_met else "below"
        return f"{relation} α = {self.alpha:.2f}"


@dataclass
class IterationRecord:
    """One post-update iterate, without a rollout or copied history."""

    controls: torch.Tensor
    loss: float
    smooth_lower: float
    hard_interval: tuple[float, float] | None
    beta: float
    control_cost: float = 0.0


@dataclass
class MPCResult:
    states: list
    applied_controls: torch.Tensor  # [steps, control_dim]
    window_plans: list[PlanResult]
    stopped_reason: str
    failure_detail: str | None = None


class Planner:
    """Optimize controls for any belief rollout and pdSTL spec.

    rollout(v) maps unconstrained parameters v to a BeliefRollout. The planner
    ascends the smooth lower bound (plus small control regularizers). With a
    threshold alpha it stops at the first iterate whose exact lower robustness
    reaches alpha; otherwise it runs max_iters steps. It returns the iterate
    with the highest exact lower robustness.
    """

    def __init__(self, dynamics, horizon, config=None):
        defaults = load_config("configs/planning.yaml")
        config = config or {}
        unknown = set(config) - set(defaults)
        if unknown:
            raise ValueError(f"unknown planner settings: {sorted(unknown)}")
        self.dyn, self.horizon = dynamics, horizon
        self.device = dynamics.device
        self.cfg = {**defaults, **config}
        self.cfg["smoothing"] = {
            **defaults["smoothing"],
            **config.get("smoothing", {}),
        }

    @property
    def control_dim(self):
        return self.dyn.B.shape[1]

    def _control_parameters(self, controls, *, margin=1e-3):
        controls = torch.as_tensor(
            controls, device=self.device, dtype=self.dyn.B.dtype
        )
        if controls.shape != (self.horizon, self.control_dim):
            raise ValueError("controls must have shape [horizon, control_dim]")
        if (
            not torch.isfinite(controls).all()
            or (controls.abs() > self.dyn.u_max).any()
        ):
            raise ValueError(
                "controls must be finite and within the control bounds"
            )
        # A finite inverse is necessary at exact saturation.
        return torch.atanh(
            (controls / self.dyn.u_max).clamp(-1 + margin, 1 - margin)
        )

    def _init_controls(self, init_guess):
        values = (
            torch.zeros(
                self.horizon,
                self.control_dim,
                device=self.device,
                dtype=self.dyn.B.dtype,
            )
            if init_guess is None
            else self._control_parameters(init_guess)
        )
        return torch.nn.Parameter(values.detach().clone())

    def _control_cost(self, controls):
        smoothness = (controls[1:] - controls[:-1]).square().sum()
        return (
            self.cfg["w_u"] * controls.square().sum()
            + self.cfg["w_du"] * smoothness
        )

    def _objective(self, controls, smooth_lower):
        return -self.cfg["w_phi"] * smooth_lower + self._control_cost(controls)

    def _applied_controls(self, prediction, parameters):
        if prediction.aux is not None and "applied_controls" in prediction.aux:
            return prediction.aux["applied_controls"]
        return self.dyn.bound_control(parameters)

    def _beta(self, iteration):
        smoothing = self.cfg["smoothing"]
        start, end = smoothing["beta_start"], smoothing["beta_end"]
        if self.cfg["max_iters"] == 1:
            return end
        return start * (end / start) ** (
            iteration / max(self.cfg["max_iters"] - 1, 1)
        )

    def _evaluate(self, rollout, parameters, spec, beta, history):
        with torch.no_grad():
            prediction = rollout(parameters)
            controls = (
                self._applied_controls(prediction, parameters).detach().clone()
            )
            smooth = spec.smooth_lower(
                prediction.belief_trajectory, beta
            ).item()
            control_cost = self._control_cost(controls).item()
            loss = self._objective(controls, smooth).item()
            interval = spec.probability_interval(prediction.belief_trajectory)
            hard_interval = tuple(interval.tolist())
            alpha = self.cfg["alpha"]
            return PlanResult(
                controls,
                prediction,
                smooth,
                hard_interval,
                list(history),
                loss,
                beta,
                control_cost=control_cost,
                alpha=alpha,
                threshold_met=(
                    None if alpha is None else hard_interval[0] >= alpha
                ),
                hard_lower_history=[hard_interval[0]],
            )

    def evaluate_controls(self, rollout, controls, *, spec):
        """Score given controls: smooth bound at beta_end and exact interval."""
        margin = torch.finfo(self.dyn.B.dtype).eps
        return self._evaluate(
            rollout,
            self._control_parameters(controls, margin=margin),
            spec,
            self.cfg["smoothing"]["beta_end"],
            [],
        )

    def optimize_window(
        self,
        rollout,
        *,
        spec,
        init_guess=None,
        verbose=False,
        on_iteration=None,
        callback_every=1,
    ):
        """Adam steps from init_guess until the exact bound reaches alpha.

        Iterate -1 is the initial guess; its exact and smooth lower bounds
        (at beta_end) are kept as `initial_hard_lower` / `initial_smooth_lower`.
        Without alpha, or if alpha is never reached, all max_iters steps run.
        on_iteration(iteration, record) observes every `callback_every`-th
        iterate and the last one.
        """
        alpha = self.cfg["alpha"]
        max_iters = self.cfg["max_iters"]
        parameters = self._init_controls(init_guess)
        initial = self._evaluate(
            rollout,
            parameters.detach().clone(),
            spec,
            self.cfg["smoothing"]["beta_end"],
            [],
        )
        started = perf_counter()
        optimizer = torch.optim.Adam([parameters], lr=self.cfg["lr"])
        candidates, history, hard_history = [], [], []
        for step in range(max_iters + 1):
            iteration = step - 1
            beta = self._beta(min(step, max_iters - 1))
            optimizer.zero_grad()
            prediction = rollout(parameters)
            lower = spec.smooth_lower(prediction.belief_trajectory, beta)
            controls = self._applied_controls(prediction, parameters)
            loss = self._objective(controls, lower)
            if not torch.isfinite(loss):
                raise ValueError("planning loss is not finite")
            with torch.no_grad():
                interval = spec.probability_interval(
                    prediction.belief_trajectory
                )
            record = IterationRecord(
                controls.detach().clone(),
                loss.item(),
                lower.item(),
                tuple(interval.tolist()),
                beta,
                self._control_cost(controls).item(),
            )
            candidates.append((iteration, record))
            hard_history.append(record.hard_interval[0])
            reached = alpha is not None and record.hard_interval[0] >= alpha
            if iteration >= 0:
                history.append(record.loss)
                observe = on_iteration is not None and (
                    iteration == 0
                    or (iteration + 1) % callback_every == 0
                    or iteration == max_iters - 1
                    or reached
                )
                if observe:
                    on_iteration(iteration, record)
                if verbose and iteration % 50 == 0:
                    logger.info(
                        "Iteration %d: loss %.6f, smooth lower %.6f, "
                        "rho_lower %.6f",
                        iteration,
                        record.loss,
                        record.smooth_lower,
                        record.hard_interval[0],
                    )
            if step == max_iters or reached:
                break
            loss.backward()
            if (
                parameters.grad is None
                or not torch.isfinite(parameters.grad).all()
            ):
                raise ValueError(
                    "rollout must supply finite gradients to controls"
                )
            optimizer.step()
        result_iteration, selected = self._select_candidate(candidates)
        result = self.evaluate_controls(rollout, selected.controls, spec=spec)
        result.loss_history = history
        result.hard_lower_history = hard_history
        result.selected_iteration = result_iteration
        result.initial_hard_lower = initial.hard_interval[0]
        result.initial_smooth_lower = initial.smooth_lower
        result.planning_time = perf_counter() - started
        return result

    @staticmethod
    def _select_candidate(candidates):
        """Highest exact lower bound; the later iterate wins a tie.

        With early stopping at alpha, this is the iterate that reached alpha.
        """
        return max(reversed(candidates), key=lambda c: c[1].hard_interval[0])

    @staticmethod
    def shift_controls(controls):
        """Drop the executed control and repeat the last one."""
        return torch.cat((controls[1:], controls[-1:])).detach().clone()

    def run_receding_horizon(
        self,
        state,
        *,
        make_rollout,
        execute,
        make_spec,
        is_done,
        max_steps,
        init_guess=None,
        verbose=False,
        on_step=None,
        on_iteration=None,
        callback_every=1,
        capture_planning_failures=False,
    ):
        """Plan a window, execute its first control, shift, and repeat.

        make_rollout, make_spec and is_done take (state, step); execute takes
        (state, control, step) and returns the next state; on_step observes.
        """
        states, applied, plans = [state], [], []
        failure_detail = None
        guess = init_guess
        reason = is_done(state, 0)
        for step in range(max_steps):
            if reason:
                break
            try:
                plan = self.optimize_window(
                    make_rollout(state, step),
                    spec=make_spec(state, step),
                    init_guess=guess,
                    verbose=verbose,
                    on_iteration=(
                        (
                            lambda iteration, record: on_iteration(
                                step, iteration, record
                            )
                        )
                        if on_iteration is not None
                        else None
                    ),
                    callback_every=callback_every,
                )
            except (RuntimeError, ValueError) as error:
                if not capture_planning_failures:
                    raise
                reason = "planning_failure"
                failure_detail = f"{type(error).__name__}: {error}"
                break
            control = plan.controls[0].clone()
            state = execute(state, control, step)
            states.append(state)
            applied.append(control)
            plans.append(plan)
            guess = self.shift_controls(plan.controls)
            reason = is_done(state, step + 1)
            if on_step is not None:
                on_step(step, state, plan)
        controls = (
            torch.stack(applied)
            if applied
            else torch.empty(
                0, self.control_dim, device=self.device, dtype=self.dyn.B.dtype
            )
        )
        return MPCResult(
            states,
            controls,
            plans,
            (reason if isinstance(reason, str) else "goal_reached")
            if reason
            else "max_steps",
            failure_detail,
        )
