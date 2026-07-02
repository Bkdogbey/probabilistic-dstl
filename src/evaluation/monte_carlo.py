import torch


def rollout_controls(
    dynamics,
    x0_mean,
    x0_cov,
    u_trace,
    *,
    num_rollouts=200,
    q_std=None,
    seed=0,
):
    """
    Roll out a fixed control sequence under stochastic process noise.

    Parameters
    ----------
    dynamics:
        Existing dynamics object.

    x0_mean:
        Tensor of shape [state_dim].

    x0_cov:
        Tensor of shape [state_dim, state_dim].

    u_trace:
        Tensor of shape [1, T, control_dim] or [T, control_dim].

    num_rollouts:
        Number of stochastic executions.

    q_std:
        Optional evaluation noise standard deviation.
        If provided, temporarily override the dynamics process noise.

    seed:
        Random seed.

    Returns
    -------
    rollouts:
        Tensor of shape [num_rollouts, T+1, state_dim].
    """
    torch.manual_seed(seed)

    device = x0_mean.device
    dtype = x0_mean.dtype

    if u_trace.ndim == 3:
        u_seq = u_trace[0]
    else:
        u_seq = u_trace

    T = u_seq.shape[0]
    state_dim = x0_mean.shape[0]

    had_q_std = hasattr(dynamics, "q_std")
    had_Q = hasattr(dynamics, "Q")
    old_q_std = getattr(dynamics, "q_std", None)
    old_Q = getattr(dynamics, "Q", None)

    if q_std is not None:
        dynamics.q_std = q_std
        dynamics.Q = torch.eye(state_dim, device=device, dtype=dtype) * (q_std ** 2)

    rollouts = []

    try:
        for _ in range(num_rollouts):
            x = x0_mean.clone()
            P = x0_cov.clone()
            traj = [x.clone()]

            for t in range(T):
                u = u_seq[t]
                pred_mean, next_cov = dynamics.step(x, P, u)
                noise = torch.distributions.MultivariateNormal(
                    torch.zeros_like(pred_mean),
                    dynamics.Q,
                ).sample()

                x = pred_mean + noise
                P = next_cov
                traj.append(x.clone())

            rollouts.append(torch.stack(traj, dim=0))
    finally:
        if q_std is not None:
            if had_q_std:
                dynamics.q_std = old_q_std
            elif hasattr(dynamics, "q_std"):
                delattr(dynamics, "q_std")

            if had_Q:
                dynamics.Q = old_Q
            elif hasattr(dynamics, "Q"):
                delattr(dynamics, "Q")

    return torch.stack(rollouts, dim=0)
