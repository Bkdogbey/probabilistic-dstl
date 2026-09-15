import torch
import torch.nn as nn

# Controlled dynamics
#
# Dynamics defines one linear step, one control convention, and one rollout;
# concrete models only set A, B, Q.


class Dynamics(nn.Module):
    """Linear controlled dynamics: x' = A x + B u,  P' = A P A^T + Q.

    Subclasses register ``A``, ``B``, ``Q`` as buffers in ``__init__``. ``Q``
    is the process noise added over one discrete step (no ``dt`` factor).
    """

    def __init__(self, dt, u_max, device="cpu"):
        super().__init__()
        self.dt = dt
        self.u_max = u_max
        self.device = device

    def bound_control(self, v):
        """Smoothly bound an unconstrained control to ``[-u_max, u_max]``.

        The one control convention: every physical control, inside a rollout
        or read externally for a cost, goes through this.
        """
        return self.u_max * torch.tanh(v)

    def step(self, x, P, u):
        """Prediction: one step of the mean/covariance. x: [D], P: [D,D], u: [control dim]."""
        x_next = self.A @ x + self.B @ u
        P_next = self.A @ P @ self.A.t() + self.Q
        return x_next, P_next

    def sample_step(self, x, P, u):
        """Simulated physical transition: X' = A x + B u + W,  W ~ N(0, Q).

        Returns the sampled next state and the predicted next covariance, so an
        executed step and a predicted step share the same A, B and Q.
        """
        x_next, P_next = self.step(x, P, u)
        noise = torch.distributions.MultivariateNormal(
            torch.zeros_like(x_next), self.Q
        ).sample()
        return x_next + noise, P_next

    def forward(self, v_sequence, x0_mean, x0_cov):
        """Roll out mean and covariance for an unconstrained control sequence.

        v_sequence: [T, control dim]   x0_mean: [D]   x0_cov: [D,D]
        Returns mean [1,T+1,D], covariance [1,T+1,D,D].
        """
        means, covs = [x0_mean], [x0_cov]
        for v in v_sequence:
            mean, cov = self.step(means[-1], covs[-1], self.bound_control(v))
            means.append(mean)
            covs.append(cov)
        return torch.stack(means).unsqueeze(0), torch.stack(covs).unsqueeze(0)


class SingleIntegrator(Dynamics):
    """Position-velocity model. State: [x, y]. Control: [vx, vy]."""

    def __init__(self, dt=0.2, u_max=1.0, q_std=0.05, device="cpu"):
        super().__init__(dt, u_max, device)
        self.register_buffer("A", torch.eye(2, device=device))
        self.register_buffer("B", dt * torch.eye(2, device=device))
        self.register_buffer("Q", torch.eye(2, device=device) * q_std**2)


class DoubleIntegrator(Dynamics):
    """Acceleration-controlled model. State: [px, py, vx, vy]. Control: [ax, ay]."""

    def __init__(self, dt=0.2, u_max=1.0, q_std=0.02, device="cpu"):
        super().__init__(dt, u_max, device)
        self.register_buffer(
            "A",
            torch.tensor(
                [
                    [1.0, 0.0, dt, 0.0],
                    [0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
            ),
        )
        self.register_buffer(
            "B",
            torch.tensor(
                [[0.5 * dt**2, 0.0], [0.0, 0.5 * dt**2], [dt, 0.0], [0.0, dt]],
                device=device,
            ),
        )
        self.register_buffer("Q", torch.eye(4, device=device) * q_std**2)
