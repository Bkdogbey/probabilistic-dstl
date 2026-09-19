"""Deterministic stlcg-style STL on the mean trajectory, for comparison.

Outputs [B, T+1, 1] signed-distance robustness in forward time."""

import numpy as np
import torch
import torch.nn as nn

from pdstl.operators import Maxish, Minish


# --- BASE CLASS ---


class DetSTL_Formula(nn.Module):
    """Base deterministic formula: BeliefTrajectory (means only) -> [B, T+1, 1] robustness."""

    def robustness_trace(self, mu, beta=None, **kwargs):
        """
        mu: [B, T+1, D] mean trajectory (forward time)
        Returns: [B, T+1, 1]
        """
        raise NotImplementedError

    def _extract_mean(self, belief_trajectory):
        means = [b.value() for b in belief_trajectory]
        return torch.stack(means, dim=1)  # [B, T+1, D]

    def forward(self, belief_trajectory, beta=None, **kwargs):
        mu = self._extract_mean(belief_trajectory)
        return self.robustness_trace(mu, beta=beta, **kwargs)


# --- TEMPORAL OPERATORS (stlcg RNN logic, forward-time interface) ---


class DetTemporalOperator(DetSTL_Formula):
    """stlcg RNN window operator, run on the reversed trace and flipped back."""

    def __init__(self, subformula, interval=None):
        super().__init__()
        self.subformula = subformula
        self.interval = interval
        self._interval = [0, np.inf] if interval is None else interval
        self.rnn_dim = 1 if not interval else interval[-1]
        if self.rnn_dim == np.inf:
            self.rnn_dim = self._interval[0]
        self.steps = 1 if not interval else interval[-1] - interval[0] + 1
        self.operation = None
        # Shift matrices for the sliding window (identical to stlcg)
        M = np.diag(np.ones(self.rnn_dim - 1), k=1)
        self.register_buffer(
            "M", torch.tensor(M, dtype=torch.float32), persistent=False
        )
        b = torch.zeros(self.rnn_dim, 1, dtype=torch.float32)
        b[-1] = 1.0
        self.register_buffer("b", b, persistent=False)

    def _initialize_rnn_cell(self, x):
        """x: [B, T+1, 1] time-reversed. Init hidden state from first (= time T) element."""
        h0 = (
            torch.ones(x.shape[0], self.rnn_dim, x.shape[2], device=x.device)
            * x[:, :1, :]
        )
        if (self._interval[1] == np.inf) and (self._interval[0] > 0):
            d0 = x[:, :1, :]
            return ((d0, h0), 0.0)
        return (h0, 0.0)

    def _rnn_cell(self, x, hc, beta=None):
        """Single RNN step. Mirrors stlcg's Always/Eventually._rnn_cell exactly."""
        h0, c = hc
        if self.interval is None:
            input_ = torch.cat([h0, x], dim=1)  # [B, rnn_dim+1, 1]
            output = self.operation(input_, beta, dim=1, keepdim=True)
            state = (output, None)
        elif (self._interval[1] == np.inf) and (self._interval[0] > 0):
            d0, h0 = h0
            dh = torch.cat([d0, h0[:, :1, :]], dim=1)  # [B, 2, 1]
            output = self.operation(dh, beta, dim=1, keepdim=True)
            state = ((output, torch.matmul(self.M, h0) + self.b * x), None)
        else:  # [a, b]
            state = (torch.matmul(self.M, h0) + self.b * x, None)
            h0x = torch.cat([h0, x], dim=1)  # [B, rnn_dim+1, 1]
            input_ = h0x[:, : self.steps, :]
            output = self.operation(input_, beta, dim=1, keepdim=True)
        return output, state

    def robustness_trace(self, mu, beta=None, **kwargs):
        # 1. Subformula trace in forward time → [B, T+1, 1]
        sub_fwd = self.subformula.robustness_trace(mu, beta=beta, **kwargs)
        # 2. Reverse → time-reversed (stlcg convention)
        sub_rev = torch.flip(sub_fwd, dims=[1])
        # 3. Run stlcg RNN
        outputs = []
        hc = self._initialize_rnn_cell(sub_rev)
        for xi in torch.split(sub_rev, 1, dim=1):
            o, hc = self._rnn_cell(xi, hc, beta=beta)
            outputs.append(o)
        out_rev = torch.cat(outputs, dim=1)  # [B, T+1, 1] reversed
        # 4. Flip back to forward time
        return torch.flip(out_rev, dims=[1])


class DetAlways(DetTemporalOperator):
    """□[a,b] φ  —  minimum robustness over [a, b]. Uses Minish."""

    def __init__(self, subformula, interval=None):
        super().__init__(subformula, interval)
        self.operation = Minish()


class DetEventually(DetTemporalOperator):
    """♢[a,b] φ  —  maximum robustness over [a, b]. Uses Maxish."""

    def __init__(self, subformula, interval=None):
        super().__init__(subformula, interval)
        self.operation = Maxish()


# --- LOGICAL OPERATORS ---


class DetAnd(DetSTL_Formula):
    """φ ∧ ψ  —  min(ρ_φ, ρ_ψ) element-wise."""

    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.operation = Minish()

    def robustness_trace(self, mu, beta=None, **kwargs):
        r1 = self.subformula1.robustness_trace(
            mu, beta=beta, **kwargs
        )  # [B, T+1, 1]
        r2 = self.subformula2.robustness_trace(mu, beta=beta, **kwargs)
        xx = torch.cat([r1, r2], dim=-1)  # [B, T+1, 2]
        return self.operation(xx, beta, dim=-1, keepdim=True)  # [B, T+1, 1]


# --- DETERMINISTIC PREDICATES (signed distance on mean trajectory) ---


class DetRectangularGoalPredicate(DetSTL_Formula):
    """
    Signed distance to being inside goal G = [x_min, x_max] × [y_min, y_max].

        ρ(t) = min(μ_x − x_min, x_max − μ_x, μ_y − y_min, y_max − μ_y)

    Positive iff mean is inside the goal rectangle.
    Mirrors RectangularGoalPredicate from environment.py.
    """

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]

    def robustness_trace(self, mu, **kwargs):
        mu_x = mu[..., 0:1]  # [B, T+1, 1]
        mu_y = mu[..., 1:2]
        stacked = torch.cat(
            [
                mu_x - self.x_min,
                self.x_max - mu_x,
                mu_y - self.y_min,
                self.y_max - mu_y,
            ],
            dim=-1,
        )  # [B, T+1, 4]
        return stacked.min(dim=-1, keepdim=True)[0]  # [B, T+1, 1]


class DetRectangularObstaclePredicate(DetSTL_Formula):
    """
    Signed distance to being outside (safe from) obstacle O = [x_min, x_max] × [y_min, y_max].

        ρ(t) = max(x_min − μ_x, μ_x − x_max, y_min − μ_y, μ_y − y_max)

    Positive iff mean is outside the obstacle (safe).
    Mirrors RectangularObstaclePredicate from environment.py.
    """

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]

    def robustness_trace(self, mu, **kwargs):
        mu_x = mu[..., 0:1]
        mu_y = mu[..., 1:2]
        stacked = torch.cat(
            [
                self.x_min - mu_x,
                mu_x - self.x_max,
                self.y_min - mu_y,
                mu_y - self.y_max,
            ],
            dim=-1,
        )
        return stacked.max(dim=-1, keepdim=True)[0]


class DetMovingRectangularObstaclePredicate(DetSTL_Formula):
    """
    Signed distance to being outside a moving rectangular obstacle.

        ρ(t) = max(x_min(t) − μ_x, μ_x − x_max(t), y_min(t) − μ_y, μ_y − y_max(t))

    x_traj, y_traj are obstacle center positions in forward time, shape [T+1].
    Positive iff mean is outside the obstacle at each time step.
    Mirrors MovingRectangularObstaclePredicate from environment.py.
    """

    def __init__(self, obs_def, device="cpu"):
        super().__init__()
        self.x_traj = torch.as_tensor(
            obs_def["x_traj"], dtype=torch.float32, device=device
        )
        self.y_traj = torch.as_tensor(
            obs_def["y_traj"], dtype=torch.float32, device=device
        )
        self.width = obs_def["width"]
        self.height = obs_def["height"]

    def robustness_trace(self, mu, **kwargs):
        x_min = self.x_traj - self.width / 2.0  # [T+1]
        x_max = self.x_traj + self.width / 2.0
        y_min = self.y_traj - self.height / 2.0
        y_max = self.y_traj + self.height / 2.0

        mu_x = mu[..., 0]  # [B, T+1]
        mu_y = mu[..., 1]

        stacked = torch.stack(
            [x_min - mu_x, mu_x - x_max, y_min - mu_y, mu_y - y_max], dim=-1
        )  # [B, T+1, 4]
        return stacked.max(dim=-1, keepdim=True)[0]  # [B, T+1, 1]


# --- SPECIFICATION BUILDER ---


def _det_obstacle(region):
    from planning.environment import MovingRectangleRegion

    if isinstance(region, MovingRectangleRegion):
        centers = region.centers
        return DetMovingRectangularObstaclePredicate(
            {
                "x_traj": centers[..., 0],
                "y_traj": centers[..., 1],
                "width": region.width,
                "height": region.height,
            }
        )
    return DetRectangularObstaclePredicate({"x": region.x, "y": region.y})


def det_get_specification(env, T, t_goal_start=0, t_constraints_start=1):
    """Deterministic mirror of Environment.get_specification on the mean trajectory."""

    def box(region):
        return {"x": region.x, "y": region.y}

    specs = []

    # 1. Goal
    if env.by_role("goal"):
        specs.append(
            DetEventually(
                DetRectangularGoalPredicate(box(env.single_region("goal"))),
                interval=[t_goal_start, T],
            )
        )

    # 2. Obstacle safety
    obs_preds = [_det_obstacle(region) for region in env.by_role("obstacle")]

    if obs_preds:
        safe_formula = obs_preds[0]
        for p in obs_preds[1:]:
            safe_formula = DetAnd(safe_formula, p)
        specs.append(
            DetAlways(safe_formula, interval=[t_constraints_start, T])
        )

    # 3. Workspace bounds
    if env.by_role("workspace"):
        specs.append(
            DetAlways(
                DetRectangularGoalPredicate(
                    box(env.single_region("workspace"))
                ),
                interval=[t_constraints_start, T],
            )
        )

    if not specs:
        raise ValueError("No constraints defined in environment.")

    combined = specs[0]
    for s in specs[1:]:
        combined = DetAnd(combined, s)

    return combined


def compare_lane_window(plan, environment, horizon):
    """Evaluate one stored lane plan against nominal road, traffic, and dwell."""
    metadata = environment.metadata
    ego = plan.rollout.aux["mean_trace"][0]
    traffic = plan.rollout.aux["traffic_mean_trace"][0]
    road = metadata["road"]
    collision = metadata["collision"]
    task = metadata["task"]
    step = metadata.get("step", 0)
    road_margin = torch.minimum(
        ego[:, 1] - road["y_min"], road["y_max"] - ego[:, 1]
    )
    separation = traffic[:, :, :2] - ego[:, None, :2]
    vehicle_margin = (
        torch.maximum(
            separation[..., 0].abs() - collision["longitudinal"],
            separation[..., 1].abs() - collision["lateral"],
        )
        .min(dim=-1)
        .values
    )
    safety = torch.minimum(road_margin.min(), vehicle_margin.min())
    target_margin = (
        task["target_tolerance"] - (ego[:, 1] - task["target_center"]).abs()
    )
    dwell = task["dwell_steps"]
    start = max(0, task["start_end_steps"][0] - step)
    end = min(task["start_end_steps"][1] - step, horizon - dwell)
    if end >= start:
        completion = torch.stack(
            [
                target_margin[t : t + dwell + 1].min()
                for t in range(start, end + 1)
            ]
        ).max()
    else:
        completion = target_margin.new_tensor(float("-inf"))
    value = torch.minimum(safety, completion)
    return {
        "pdstl_probability_interval": plan.hard_interval,
        "deterministic_signed_distance": float(value.detach()),
    }
