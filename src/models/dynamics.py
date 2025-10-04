import numpy as np


def control_input(t):
    """Control input function u(t)."""
    return -0.5  # constant downward velocity


def first_order_system(a, b, g, q, mu, P, t):
    """Propagate the belief state (mu, P) through one time step."""
    mean_trace = np.zeros(len(t))
    var_trace = np.zeros(len(t))

    mean_trace[0] = mu
    var_trace[0] = P
    # TODO: Instead of writing the dynamics, use established python packages
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        u = control_input(t[i - 1])  # control input at time t[i-1]

        Phi = np.exp(a * dt)
        int_u = dt * b * u  # integral of b*u from t[i-1] to t[i]
        mean_trace[i] = Phi * mean_trace[i - 1] + int_u

        # Variance update
        var_trace[i] = (Phi**2) * var_trace[i - 1] + (g**2) * dt + q * dt
    return mean_trace, var_trace
