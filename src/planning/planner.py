"""Gradient planning on a belief rollout.

One planning operation, one result type:

    controls -> belief rollout -> pdSTL lower bound -> backward -> Adam step

The smooth lower score is an optimization surrogate and supplies every gradient. The exact
hard evaluation is the semantic pdSTL interval: it monitors, selects the returned candidate
and reports the outcome, and it never backpropagates.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim

from models.rollouts import BeliefRollout, gaussian_rollout
from planning import log_utils


@dataclass(frozen=True)
class OptimizationRecord:
    """One iteration, scalars only -- no rollout and no autograd graph."""

    iteration: int
    beta: float | None
    smooth_lower: float
    hard_lower: float
    loss: float


@dataclass
class PlanResult:
    """One solved planning window. Every derived quantity comes from these fields."""

    controls: torch.Tensor                  # [H, m] bounded controls
    rollout: BeliefRollout                  # beliefs, plus aux mean_trace / cov_trace
    hard_interval: torch.Tensor             # [2] exact pdSTL interval at origin 0
    history: list[OptimizationRecord]
    best_iteration: int

    @property
    def hard_lower(self):
        return float(self.hard_interval[0])

    @property
    def hard_upper(self):
        return float(self.hard_interval[1])


@dataclass
class PlanConfig:
    """Optimizer settings, from the `optimizer:` block of a scenario file."""

    learning_rate: float = 0.03
    max_iterations: int = 500
    probability_target: float = 0.80
    convergence_patience: int = 15

    smoothing_enabled: bool = True
    beta_start: float = 1.0
    beta_end: float = 20.0

    pdstl_weight: float = 1.0
    control_effort_weight: float = 0.001
    control_smoothness_weight: float = 0.01
    terminal_goal_weight: float = 0.0

    extras: dict = field(default_factory=dict)  # passed to a scenario's extra_loss hook

    @classmethod
    def from_config(cls, config):
        config = dict(config or {})
        smoothing = dict(config.get("smoothing") or {})
        loss = dict(config.get("loss") or {})
        known = {
            "learning_rate", "max_iterations", "probability_target", "convergence_patience",
        }
        return cls(
            **{k: config[k] for k in known & set(config)},
            smoothing_enabled=smoothing.get("enabled", cls.smoothing_enabled),
            beta_start=smoothing.get("beta_start", cls.beta_start),
            beta_end=smoothing.get("beta_end", cls.beta_end),
            pdstl_weight=loss.get("pdstl_weight", cls.pdstl_weight),
            control_effort_weight=loss.get("control_effort_weight", cls.control_effort_weight),
            control_smoothness_weight=loss.get(
                "control_smoothness_weight", cls.control_smoothness_weight
            ),
            terminal_goal_weight=loss.get("terminal_goal_weight", cls.terminal_goal_weight),
            extras=config,
        )


class Planner:
    """Solves one finite-horizon stochastic pdSTL optimization problem."""

    # atanh of a saturated control lands in the flat tail of tanh, so a warm start is
    # pulled back inside the bound; a replay is not, because it must round-trip.
    WARM_START_MARGIN = 1e-2
    REPLAY_MARGIN = 1e-6

    def __init__(self, dynamics, environment, horizon, config=None):
        self.dynamics = dynamics
        self.environment = environment
        self.horizon = horizon
        self.device = dynamics.device
        self.config = config if isinstance(config, PlanConfig) else PlanConfig.from_config(config)

    # --- Control parameterization ----------------------------------------------

    @property
    def control_dim(self):
        return self.dynamics.B.shape[1]

    def control_parameters(self, controls, margin=REPLAY_MARGIN):
        """Inverse of bound_control, clipped to within `margin` of the bound."""
        normalized = torch.clamp(
            controls / (self.dynamics.u_max + 1e-6), -1.0 + margin, 1.0 - margin
        )
        return 0.5 * torch.log((1 + normalized) / (1 - normalized))

    def _initial_parameters(self, initial_controls):
        if initial_controls is None:
            return torch.zeros(self.horizon, self.control_dim, device=self.device)
        controls = torch.as_tensor(initial_controls, device=self.device, dtype=torch.float32)
        return self.control_parameters(controls, margin=self.WARM_START_MARGIN)

    # --- Objective --------------------------------------------------------------

    def _control_effort(self, controls):
        return torch.sum(controls**2)

    def _control_smoothness(self, controls):
        return torch.sum(controls[0] ** 2) + torch.sum((controls[1:] - controls[:-1]) ** 2)

    def _terminal_goal_cost(self, nominal_trace):
        goal = getattr(self.environment, "goal", None)
        if goal is None or nominal_trace is None:
            return torch.zeros((), device=self.device)
        center = torch.tensor(
            [sum(goal["x"]) / 2.0, sum(goal["y"]) / 2.0], device=self.device
        )
        return torch.sum((nominal_trace[:, -1, :2] - center) ** 2)

    def objective(self, *, smooth_lower, controls, rollout, extra_loss=None):
        """J = -w_phi * smooth_lower + w_u J_u + w_du J_du + w_g J_g (+ scenario extras)."""
        cfg = self.config
        loss = -cfg.pdstl_weight * smooth_lower
        loss = loss + cfg.control_effort_weight * self._control_effort(controls)
        loss = loss + cfg.control_smoothness_weight * self._control_smoothness(controls)
        if cfg.terminal_goal_weight:
            loss = loss + cfg.terminal_goal_weight * self._terminal_goal_cost(rollout.nominal_trace)

        if extra_loss is not None and rollout.nominal_trace is not None:
            extra = extra_loss(rollout.nominal_trace, cfg.extras)
            if extra is not None:
                loss = loss + extra
        return loss

    # --- Smoothing schedule -----------------------------------------------------

    def beta_schedule(self, iteration):
        """Geometric beta_start -> beta_end over max_iterations, or None when smoothing is off."""
        cfg = self.config
        if not cfg.smoothing_enabled:
            return None
        span = max(cfg.max_iterations - 1, 1)
        return cfg.beta_start * (cfg.beta_end / cfg.beta_start) ** (iteration / span)

    # --- The optimizer ----------------------------------------------------------

    def solve(
        self, initial_mean, initial_covariance, initial_controls=None, specification=None,
        extra_loss=None, on_iteration=None,
    ):
        """Optimize one window from (mean, covariance); returns a PlanResult.

        `specification` and `extra_loss` are passed in rather than read off a mutable
        `self.environment`, so a receding-horizon window never mutates the planner.
        `on_iteration(record, controls)` observes each iterate and cannot change the result.
        """
        rollout = gaussian_rollout(self.dynamics, initial_mean, initial_covariance)
        if specification is None:
            specification = self.environment.specification(self.horizon)
        if extra_loss is None and self.environment is not None:
            extra_loss = getattr(self.environment, "extra_loss", None)

        parameters, history, best_iteration = self._optimize_controls(
            rollout, specification, initial_controls, extra_loss, on_iteration
        )

        # Replay the checkpointed controls and recompute the exact interval.
        controls = self.dynamics.bound_control(parameters).detach()
        predicted = rollout(self.control_parameters(controls))
        with torch.no_grad():
            hard_interval = specification(predicted.belief_trajectory, scale=-1)[0, 0].clone()

        return PlanResult(
            controls=controls,
            rollout=predicted.detach_diagnostics(),
            hard_interval=hard_interval,
            history=history,
            best_iteration=best_iteration,
        )

    def _optimize_controls(self, rollout, specification, initial_controls, extra_loss=None,
                           on_iteration=None):
        """Adam on the smooth lower score; checkpoint on the exact hard lower score."""
        cfg = self.config
        parameters = nn.Parameter(self._initial_parameters(initial_controls))
        optimizer = optim.Adam([parameters], lr=cfg.learning_rate)

        history, best, at_target = [], None, 0

        for iteration in range(cfg.max_iterations):
            optimizer.zero_grad()

            controls = self.dynamics.bound_control(parameters)
            predicted = rollout(parameters)
            beta = self.beta_schedule(iteration)

            smooth_lower = specification(predicted.belief_trajectory, scale=beta or -1)[0, 0, 0]
            loss = self.objective(
                smooth_lower=smooth_lower, controls=controls, rollout=predicted,
                extra_loss=extra_loss,
            )

            # Hard evaluation before the step, so the checkpoint belongs to these controls.
            with torch.no_grad():
                hard_interval = specification(predicted.belief_trajectory, scale=-1)[0, 0]
                hard_lower = float(hard_interval[0])
                smooth_value, loss_value = float(smooth_lower), float(loss)

            record = OptimizationRecord(
                iteration=iteration, beta=beta,
                smooth_lower=smooth_value, hard_lower=hard_lower, loss=loss_value,
            )
            history.append(record)
            if on_iteration is not None:
                on_iteration(record, controls.detach().clone())
            # Exact hard lower first; ties go to the later iterate.
            #
            # Neither obvious alternative works as a tie-break. Control effort alone returns
            # the do-nothing plan whenever the hard score is flat at zero -- which is exactly
            # the early, far-from-goal regime -- because zero controls are the cheapest. The
            # smooth score is not comparable across iterations either, since beta annealing
            # changes the function being evaluated, and it peaks at iteration 0. Effort is
            # already priced into the loss, so it does not need a second vote here.
            score = (hard_lower, iteration)
            if best is None or score > best[1]:
                best = (parameters.detach().clone(), score, iteration)

            loss.backward()
            optimizer.step()

            at_target = at_target + 1 if hard_lower >= cfg.probability_target else 0
            if at_target >= cfg.convergence_patience:
                log_utils._log.debug(
                    f"converged at iteration {iteration}, hard lower {hard_lower:.4f}"
                )
                break

        return best[0], history, best[2]
