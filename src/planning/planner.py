from typing import NamedTuple

import torch
import torch.nn as nn
import torch.optim as optim

from models.rollouts import BeliefRollout, gaussian_rollout
from planning import log_utils
from utils import load_config


class PlanCandidate(NamedTuple):
    """One control sequence evaluated with smooth and hard pdSTL semantics."""

    controls: torch.Tensor
    rollout: BeliefRollout
    smooth_lower: float  # differentiable lower semantics at beta (hard if beta=None)
    hard_interval: tuple  # unsmoothed stochastic robustness [R_lower, R_upper]
    control_cost: float
    objective: float
    beta: float | None = None
    iteration: int | None = None

    @property
    def hard_lower(self):
        return self.hard_interval[0]

    @property
    def hard_upper(self):
        return self.hard_interval[1]


class Planner:
    """Gradient-based pdSTL planner over any belief rollout."""

    def __init__(self, dynamics, environment, T, config=None):
        self.dyn = dynamics
        self.env = environment
        self.T = T
        self.device = dynamics.device

        defaults, config = load_config("configs/planning.yaml"), config or {}
        self.cfg = {**defaults, **config}
        self.cfg["smoothing"] = {**defaults["smoothing"], **config.get("smoothing", {})}

    # Warm starts trade round-trip fidelity for gradient: atanh of a saturated control
    # lands in the flat tail of tanh, so a warm start is pulled back inside the bound.
    WARM_START_MARGIN = 1e-2
    REPLAY_MARGIN = 1e-6

    def _control_parameters(self, controls, margin=REPLAY_MARGIN):
        """Inverse of bound_control, clipped to within margin of the bound."""
        u_norm = torch.clamp(controls / (self.dyn.u_max + 1e-6), -1.0 + margin, 1.0 - margin)
        return 0.5 * torch.log((1 + u_norm) / (1 - u_norm))

    def _init_controls(self, init_guess):
        if init_guess is not None:
            v_init = self._control_parameters(init_guess, margin=self.WARM_START_MARGIN)
            return nn.Parameter(v_init.to(self.device), requires_grad=True)
        offset = torch.zeros(self._control_dim, device=self.device)
        offset[0] = 0.5
        return nn.Parameter(
            torch.randn(self.T, self._control_dim, device=self.device) * 0.1 + offset,
            requires_grad=True,
        )

    @property
    def _control_dim(self):
        return self.dyn.B.shape[1]

    # --- Optional shaping heuristics (legacy; off in the pdSTL demos) -----------

    def _goal_dist_loss(self, mean_trace):
        """Squared distance from final position to goal centre."""
        if self.env.goal is None:
            return torch.tensor(0.0, device=self.device)
        gx = sum(self.env.goal["x"]) / 2.0
        gy = sum(self.env.goal["y"]) / 2.0
        goal_center = torch.tensor([[gx, gy]], device=self.device)
        return torch.sum((mean_trace[:, -1, :2] - goal_center) ** 2)

    def _obs_repulsion_loss(self, mean_trace):
        """Penalise trajectory points that are too close to obstacle centres."""
        loss = torch.tensor(0.0, device=self.device)
        margin = self.cfg["obs_margin"]

        for obs in self.env.obstacles:
            cx = (obs["x"][0] + obs["x"][1]) / 2.0
            cy = (obs["y"][0] + obs["y"][1]) / 2.0
            center = torch.tensor([[cx, cy]], device=self.device)
            radius = max(obs["x"][1] - obs["x"][0], obs["y"][1] - obs["y"][0]) / 2.0 + margin
            dists = torch.norm(mean_trace[:, :, :2] - center, dim=2)
            loss = loss + torch.sum(torch.relu(radius - dists) ** 2)

        for obs in self.env.circle_obstacles:
            center = torch.tensor([obs["center"]], device=self.device)
            radius = obs["radius"] + margin
            dists = torch.norm(mean_trace[:, :, :2] - center, dim=2)
            loss = loss + torch.sum(torch.relu(radius - dists) ** 2)

        for obs in self.env.moving_obstacles:
            ox = torch.as_tensor(obs["x_traj"], device=self.device)
            oy = torch.as_tensor(obs["y_traj"], device=self.device)
            centers = torch.stack([ox, oy], dim=1).unsqueeze(0)  # [1, T+1, 2]
            radius = max(obs["width"], obs["height"]) / 2.0 + margin
            dists = torch.norm(mean_trace[:, :, :2] - centers, dim=2)
            loss = loss + torch.sum(torch.relu(radius - dists) ** 2)

        return loss

    def _visit_loss(self, mean_trace):
        """Pull trajectory towards visit regions (Eventually semantics)."""
        loss = torch.tensor(0.0, device=self.device)
        for region in self.env.visit_regions:
            vx = (region["x"][0] + region["x"][1]) / 2.0
            vy = (region["y"][0] + region["y"][1]) / 2.0
            v_center = torch.tensor([[vx, vy]], device=self.device)
            dists_sq = torch.sum((mean_trace[:, :, :2] - v_center) ** 2, dim=2)
            min_dist_sq, _ = torch.min(dists_sq, dim=1)
            loss = loss + torch.sum(min_dist_sq)
        return loss

    def _control_cost(self, controls):
        """w_u |u|^2 + w_du (|du|^2 + |u_0|^2)."""
        smoothness = torch.sum((controls[1:] - controls[:-1]) ** 2) + torch.sum(controls[0] ** 2)
        return self.cfg["w_u"] * torch.sum(controls**2) + self.cfg["w_du"] * smoothness

    def _objective(self, nominal_trace, controls, smooth_lower):
        """J = -w_phi * smooth_lower + control cost + enabled shaping."""
        objective = -self.cfg["w_phi"] * smooth_lower + self._control_cost(controls)
        shaping = (
            ("w_dist", self._goal_dist_loss),
            ("w_obs", self._obs_repulsion_loss),
            ("w_visit", self._visit_loss),
        )
        for key, term in shaping:
            if self.cfg[key]:
                if nominal_trace is None:
                    raise ValueError(
                        f"{key} shaping requires a rollout nominal_trace"
                    )
                objective = objective + self.cfg[key] * term(nominal_trace)
        return objective

    def _beta(self, k):
        """Smoothing beta at iteration k (geometric from beta_start to beta_end), or None if off."""
        smoothing = self.cfg["smoothing"]
        if not smoothing["enabled"]:
            return None
        start, end = smoothing["beta_start"], smoothing["beta_end"]
        return start * (end / start) ** (k / max(self.cfg["max_iters"] - 1, 1))

    def _scores(self, phi, belief_trajectory, beta=None):
        """Differentiable lower semantics and detached hard diagnostic interval.

        With smoothing enabled, only the smooth lower supplies gradients. The
        legacy beta=None mode differentiates the hard lower instead.
        """
        if beta is None:
            hard_interval = phi(belief_trajectory, scale=-1)[0, 0]
            return hard_interval[0], hard_interval.detach()
        smooth_lower = phi(belief_trajectory, scale=beta)[0, 0, 0]
        with torch.no_grad():
            hard_interval = phi(belief_trajectory, scale=-1)[0, 0]
        return smooth_lower, hard_interval

    def _candidate(self, phi, rollout, v, beta=None):
        """Score parameters v; returns (PlanCandidate, objective with graph)."""
        predicted = rollout(v)
        controls = self.dyn.bound_control(v)
        smooth_lower, hard_interval = self._scores(phi, predicted.belief_trajectory, beta)
        objective = self._objective(predicted.nominal_trace, controls, smooth_lower)

        candidate = PlanCandidate(
            controls=controls.detach().clone(),
            rollout=predicted.detach_diagnostics(),
            smooth_lower=smooth_lower.item(),
            hard_interval=tuple(hard_interval.tolist()),
            control_cost=self._control_cost(controls).item(),
            objective=objective.item(),
            beta=beta,
        )
        return candidate, objective

    def evaluate_controls(self, rollout, controls, *, spec):
        """Evaluate controls [T,m]; smooth lower semantics use the final beta."""
        v = self._control_parameters(controls.to(self.device))
        candidate, _ = self._candidate(spec, rollout, v, self._beta(self.cfg["max_iters"] - 1))
        return candidate

    @staticmethod
    def _better(candidate, best):
        """Checkpoint by hard lower robustness, breaking ties by control cost."""
        if best is None:
            return True
        return (candidate.hard_lower, -candidate.control_cost) > (best.hard_lower, -best.control_cost)

    def optimize_window(
        self, rollout, *, spec=None, env=None, init_guess=None, verbose=False,
        on_iteration=None,
    ):
        """Descend the smooth objective; checkpoint by hard lower robustness.

        Hard semantics monitor/checkpoint/stop; they do not supply smooth-mode
        gradients. Returns (best candidate, objective history).
        on_iteration(k, candidate) only observes.
        """
        saved_env = self.env
        if env is not None:
            self.env = env
        phi = spec if spec is not None else self.env.get_specification(self.T)

        v = self._init_controls(init_guess)
        optimizer = optim.Adam([v], lr=self.cfg["lr"])
        best, history, converged_iters = None, [], 0

        if verbose:
            log_utils._log.info(f"Starting optimisation (max iters: {self.cfg['max_iters']})")

        for k in range(self.cfg["max_iters"]):
            optimizer.zero_grad()
            beta = self._beta(k)
            candidate, objective = self._candidate(phi, rollout, v, beta)
            candidate = candidate._replace(iteration=k)
            history.append(candidate.objective)
            if on_iteration is not None:
                on_iteration(k, candidate)
            if self._better(candidate, best):
                best = candidate

            objective.backward()
            optimizer.step()

            if candidate.hard_lower >= self.cfg["alpha"]:
                converged_iters += 1
                if converged_iters >= self.cfg["converge_patience"]:
                    if verbose:
                        log_utils._log.info(f"Converged at iter {k}. Hard lower: {candidate.hard_lower:.4f}")
                    break
            else:
                converged_iters = 0

            # While beta anneals the objective itself moves, so the plateau test waits for a fixed beta.
            plateau = k > self.cfg["min_iters"] and beta == self._beta(k - 1)
            if plateau and abs(history[-2] - history[-1]) < self.cfg["loss_tol"]:
                if verbose:
                    log_utils._log.info(f"Loss converged at iter {k}.")
                break

            if verbose and k % 50 == 0:
                log_utils._log.info(
                    f"Iter {k:03d} | Objective: {candidate.objective:.4f} | "
                    f"Smooth lower: {candidate.smooth_lower:.4f} | Hard lower: {candidate.hard_lower:.4f}"
                )

        self.env = saved_env
        return best, history

    def run_receding_horizon(
        self, state, *, make_rollout, execute, is_done, spec, max_steps, init_guess=None
    ):
        """Plan, execute the first control, observe, replan until is_done or max_steps."""
        states, applied, candidates, warm_starts = [state], [], [], []
        guess = init_guess
        while not is_done(state) and len(applied) < max_steps:
            best, _ = self.optimize_window(make_rollout(state), spec=spec, init_guess=guess)
            warm_starts.append(guess)
            candidates.append(best)
            state = execute(state, best.controls[0])
            states.append(state)
            applied.append(best.controls[0])
            guess = self._shift_controls(best.controls)

        return {
            "states": states,
            "u_trace": self._stack_controls(applied),
            "candidates": candidates,
            "warm_starts": warm_starts,
            "stopped_reason": "goal_reached" if is_done(state) else "max_steps",
        }

    def _empty_u_trace(self):
        return torch.empty(1, 0, self._control_dim, device=self.device)

    def _stack_controls(self, controls):
        return torch.stack(controls).unsqueeze(0) if controls else self._empty_u_trace()

    def _shift_controls(self, prev_u_sol):
        if prev_u_sol is None:
            return None
        return torch.cat([prev_u_sol[1:], prev_u_sol[-1:]], dim=0)

    # --- Legacy environment scenarios (single shot, MPC, lane change) -------------

    def _goal_center(self, env):
        if env.goal is None:
            return None
        gx, gy = env.goal["x"], env.goal["y"]
        return torch.tensor([(gx[0] + gx[1]) / 2, (gy[0] + gy[1]) / 2], device=self.device)

    def _log_lane_change_step(self, step, curr_mean, best_p):
        obs_pos = self.env.moving_obstacle_position(step)
        if obs_pos is None or step % 5:
            return
        obs = torch.as_tensor(obs_pos, device=self.device, dtype=curr_mean.dtype)
        dist = torch.linalg.norm(curr_mean[:2] - obs).item()
        log_utils.log_lane_step(step, curr_mean.detach().cpu().numpy(), obs_pos[0], dist, best_p)

    def _lane_change_success(self, curr_mean, success_counter):
        if self.env.success is None:
            return success_counter, False
        success = self.env.success
        success_counter = success_counter + 1 if success["y_min"] <= curr_mean[1].item() <= success["y_max"] else 0
        return success_counter, success_counter >= success["consecutive_steps"]

    def _run_mpc(self, x0_mean, x0_cov, *, step_callback=None):
        """MPC with sampled execution: T_SIM fixed steps (optionally lane change) or MAX_STEPS to goal."""
        fixed = "T_SIM" in self.cfg
        lane_change = self.cfg.get("mpc_mode") == "lane_change"
        goal_center = None if fixed else self._goal_center(self.env)
        stopped_reason = "T_SIM" if fixed else "MAX_STEPS"

        mean, cov = x0_mean, x0_cov
        means, covs, controls, p_sat, losses, plans = [mean], [cov], [], [], [], []
        prev_u, success_counter = None, 0

        for step in range(self.cfg["T_SIM"] if fixed else self.cfg["MAX_STEPS"]):
            dist = None
            if goal_center is not None:
                dist = torch.norm(mean[:2] - goal_center)
                if dist < self.cfg.get("goal_reached_dist", 1.0):
                    stopped_reason = "goal_reached"
                    log_utils.log_goal_reached(step)
                    break

            env = self.env.make_local_lane_change_window(step, mean, self.cfg) if lane_change else None
            best, history = self.optimize_window(
                gaussian_rollout(self.dyn, mean, cov), env=env, init_guess=self._shift_controls(prev_u)
            )
            prev_u, plan = best.controls, best.rollout.aux["mean_trace"]
            plans.append(plan)
            p_sat.append(best.hard_lower)
            losses.append(history[-1] if history else 0.0)

            mean, cov = self.dyn.sample_step(mean, cov, best.controls[0])
            means.append(mean)
            covs.append(cov)
            controls.append(best.controls[0])

            if step_callback is not None:
                step_callback(step, mean, cov, plan, best.hard_lower)

            if lane_change:
                self._log_lane_change_step(step, mean, best.hard_lower)
                success_counter, done = self._lane_change_success(mean, success_counter)
                if done:
                    stopped_reason = "lane_change_success"
                    log_utils.log_lane_change_done(self.env.label, step)
                    break
            elif not fixed:
                distance = dist.item() if dist is not None else 0.0
                log_utils.log_mpc_step(step, mean.cpu().numpy(), distance, best.hard_lower)

        return self._pack_result(
            torch.stack(means).unsqueeze(0), torch.stack(covs).unsqueeze(0), self._stack_controls(controls),
            p_sat, losses, plans, mode="mpc_fixed" if fixed else "mpc_goal", stopped_reason=stopped_reason,
        )

    def _pack_result(self, mean_trace, cov_trace, u_trace, p_sat_trace, loss_trace, all_plans, *, mode, stopped_reason):
        return {
            "mean_trace": mean_trace,
            "cov_trace": cov_trace,
            "u_trace": u_trace,
            "p_sat_trace": p_sat_trace,
            "loss_trace": loss_trace,
            "history": loss_trace,
            "all_plans": all_plans,
            "best_p": max(p_sat_trace) if p_sat_trace else 0.0,
            "mode": mode,
            "stopped_reason": stopped_reason,
        }

    def solve(self, x0_mean, x0_cov, *, verbose=True, step_callback=None):
        """Legacy entry: MPC if the config has T_SIM or MAX_STEPS, else one window."""
        if "T_SIM" in self.cfg or "MAX_STEPS" in self.cfg:
            return self._run_mpc(x0_mean, x0_cov, step_callback=step_callback)
        best, history = self.optimize_window(gaussian_rollout(self.dyn, x0_mean, x0_cov), verbose=verbose)
        return self._pack_result(
            best.rollout.aux["mean_trace"], best.rollout.aux["cov_trace"], best.controls,
            [best.hard_lower], history, [], mode="single_shot", stopped_reason="optimized",
        )
