import torch
import torch.nn as nn


class Dynamics(nn.Module):
    """Linear dynamics x' = A x + B u + W, W ~ N(0, Q); subclasses set A, B, Q."""

    def __init__(self, dt, u_max, device="cpu"):
        super().__init__()
        self.dt = dt
        self.u_max = u_max
        self.device = device

    def bound_control(self, v):
        """Map an unconstrained parameter to a control in [-u_max, u_max]."""
        return self.u_max * torch.tanh(v)

    def step(self, x, P, u):
        """Predicted mean and covariance after one step."""
        return self.A @ x + self.B @ u, self.A @ P @ self.A.t() + self.Q

    def sample_step(self, x, P, u):
        """Simulated step: sampled next state and predicted covariance."""
        x_next, P_next = self.step(x, P, u)
        noise = torch.distributions.MultivariateNormal(
            torch.zeros_like(x_next), self.Q
        ).sample()
        return x_next + noise, P_next

    def forward(self, v_sequence, x0_mean, x0_cov):
        """Roll out v [T, m] from (x0_mean, x0_cov); returns mean [1,T+1,D], cov [1,T+1,D,D]."""
        means, covs = [x0_mean], [x0_cov]
        for v in v_sequence:
            mean, cov = self.step(means[-1], covs[-1], self.bound_control(v))
            means.append(mean)
            covs.append(cov)
        return torch.stack(means).unsqueeze(0), torch.stack(covs).unsqueeze(0)


class SingleIntegrator(Dynamics):
    """Velocity control in R^D: A = I, B = dt I, Q = q_std^2 I."""

    def __init__(
        self, dt=0.2, u_max=1.0, q_std=0.05, device="cpu", state_dim=2
    ):
        super().__init__(dt, u_max, device)
        eye = torch.eye(state_dim, device=device)
        self.register_buffer("A", eye.clone())
        self.register_buffer("B", dt * eye)
        self.register_buffer("Q", q_std**2 * eye)


class DoubleIntegrator(Dynamics):
    """Acceleration control. State [px, py, vx, vy], control [ax, ay]."""

    def __init__(self, dt=0.2, u_max=1.0, q_std=0.02, device="cpu"):
        super().__init__(dt, u_max, device)
        A = torch.eye(4, device=device)
        A[0, 2] = A[1, 3] = dt
        B = torch.zeros(4, 2, device=device)
        B[0, 0] = B[1, 1] = 0.5 * dt**2
        B[2, 0] = B[3, 1] = dt
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("Q", q_std**2 * torch.eye(4, device=device))
