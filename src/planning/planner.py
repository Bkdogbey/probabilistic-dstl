"""One optimizer and one receding-horizon loop over caller-supplied belief rollouts."""

import logging
import math
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
    smooth_history: list[float] = field(default_factory=list)
    hard_lower_history: list[float] = field(default_factory=list)


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
    """Optimize controls without knowing how the rollout represents uncertainty.

    rollout(v) returns a BeliefRollout; dynamics supplies bounds, device, dtype
    and control dimension. Only the smooth lower score is optimized.
    """

    def __init__(self, dynamics, horizon, config=None):
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon < 1
        ):
            raise ValueError("horizon must be a positive integer")
        self.dyn, self.horizon = dynamics, horizon
        self.device = dynamics.device
        defaults = load_config("configs/planning.yaml")
        config = config or {}
        unknown = set(config) - set(defaults)
        if unknown:
            raise ValueError(f"unknown planner settings: {sorted(unknown)}")
        self.cfg = {**defaults, **config}
        self.cfg["smoothing"] = {
            **defaults["smoothing"],
            **config.get("smoothing", {}),
        }
        unknown_smoothing = set(self.cfg["smoothing"]) - set(
            defaults["smoothing"]
        )
        if unknown_smoothing:
            raise ValueError(
                f"unknown smoothing settings: {sorted(unknown_smoothing)}"
            )
        if self.cfg["max_iters"] < 1:
            raise ValueError("max_iters must be positive")
        alpha = self.cfg["alpha"]
        if alpha is not None and (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or not 0 <= alpha <= 1
        ):
            raise ValueError("alpha must be null or a finite value in [0, 1]")
        if (
            isinstance(self.cfg["converge_patience"], bool)
            or not isinstance(self.cfg["converge_patience"], int)
            or self.cfg["converge_patience"] < 1
        ):
            raise ValueError("converge_patience must be a positive integer")
        tolerance = self.cfg["candidate_tolerance"]
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError(
                "candidate_tolerance must be a finite nonnegative number"
            )
        if any(
            self.cfg["smoothing"][key] <= 0
            for key in ("beta_start", "beta_end")
        ):
            raise ValueError("smoothing beta must be positive")

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
            loss = -self.cfg["w_phi"] * smooth + control_cost
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
                smooth_history=[smooth],
                hard_lower_history=[hard_interval[0]],
            )

    def evaluate_controls(self, rollout, controls, *, spec):
        """Replay physical controls using final smoothing and exact reporting semantics."""
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
        """Optimize a smooth score and select with the configured hard threshold.

        The initial controls and every post-update iterate are evaluated as
        internally consistent candidates. An optional observer receives
        ``(iteration, IterationRecord)`` for sampled post-update candidates.
        """
        if (
            isinstance(callback_every, bool)
            or not isinstance(callback_every, int)
            or callback_every < 1
        ):
            raise ValueError("callback_every must be a positive integer")
        parameters = self._init_controls(init_guess)
        started = perf_counter()
        optimizer = torch.optim.Adam([parameters], lr=self.cfg["lr"])
        history, patience = [], 0
        initial = self._evaluate(
            rollout, parameters, spec, self._beta(0), history
        )
        candidates = [(-1, initial)]
        smooth_history = [initial.smooth_lower]
        hard_history = [initial.hard_interval[0]]
        for iteration in range(self.cfg["max_iters"]):
            beta = self._beta(iteration)
            optimizer.zero_grad()
            prediction = rollout(parameters)
            lower = spec.smooth_lower(prediction.belief_trajectory, beta)
            loss = self._objective(
                self._applied_controls(prediction, parameters), lower
            )
            if not torch.isfinite(loss):
                raise ValueError("planning loss is not finite")
            loss.backward()
            if (
                parameters.grad is None
                or not torch.isfinite(parameters.grad).all()
            ):
                raise ValueError(
                    "rollout must supply finite gradients to controls"
                )
            optimizer.step()
            candidate = self._evaluate(
                rollout, parameters, spec, beta, history
            )
            if not math.isfinite(candidate.final_loss):
                raise ValueError("planning loss is not finite after update")
            candidates.append((iteration, candidate))
            history.append(candidate.final_loss)
            smooth_history.append(candidate.smooth_lower)
            hard_history.append(candidate.hard_interval[0])
            threshold = self.cfg["alpha"]
            observe = on_iteration is not None and (
                iteration == 0
                or (iteration + 1) % callback_every == 0
                or iteration == self.cfg["max_iters"] - 1
            )
            patience = (
                patience + 1
                if threshold is not None
                and candidate.hard_interval[0] >= threshold
                else 0
            )
            stopping = (
                threshold is not None
                and patience >= self.cfg["converge_patience"]
            )
            if on_iteration is not None and (observe or stopping):
                on_iteration(
                    iteration,
                    IterationRecord(
                        candidate.controls.detach().clone(),
                        candidate.final_loss,
                        candidate.smooth_lower,
                        candidate.hard_interval,
                        beta,
                        candidate.control_cost,
                    ),
                )
            if stopping:
                break
            if verbose and iteration % 50 == 0:
                logger.info(
                    "Iteration %d: loss %.6f, smooth lower %.6f",
                    iteration,
                    candidate.final_loss,
                    candidate.smooth_lower,
                )
        result_iteration, result = self._select_candidate(candidates)
        result.loss_history = history
        result.smooth_history = smooth_history
        result.hard_lower_history = hard_history
        result.selected_iteration = result_iteration
        result.planning_time = perf_counter() - started
        return result

    def _select_candidate(self, candidates):
        """Select a candidate by hard feasibility, then control cost."""
        alpha = self.cfg["alpha"]
        if alpha is None:
            return candidates[-1]

        feasible = [
            item for item in candidates if item[1].hard_interval[0] >= alpha
        ]
        if feasible:
            return min(feasible, key=lambda item: item[1].control_cost)

        tolerance = self.cfg["candidate_tolerance"]
        best = candidates[0]
        for candidate in candidates[1:]:
            score = candidate[1].hard_interval[0]
            best_score = best[1].hard_interval[0]
            if score > best_score + tolerance or (
                abs(score - best_score) <= tolerance
                and candidate[1].control_cost < best[1].control_cost
            ):
                best = candidate
        return best

    @staticmethod
    def shift_controls(controls):
        """Discard the executed control and hold the last control at the horizon tail."""
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
        """Callbacks use (state, step); execute additionally receives physical u_0.

        execute returns a new state rather than mutating recorded states.
        is_done is pure. on_step(step, updated_state, plan) only observes.
        """
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps < 0
        ):
            raise ValueError("max_steps must be a nonnegative integer")
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
