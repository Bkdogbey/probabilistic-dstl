import numpy as np


def control_input(t):
    """Control input function u(t)."""
    return -0.5  # constant downward velocity


def sinusoidial_input(t):
    """A sinusoidal control input function u(t)."""
    return 15 * np.sin(2 * np.pi * t / 5) # Sinusoidal input with period 5s and amplitude 15


def first_order_system(a, b, g, q, mu, P, t, control_func=control_input):
    """Propagate the belief state (mu, P) through one time step."""
    mean_trace = np.zeros(len(t))
    var_trace = np.zeros(len(t))

    mean_trace[0] = mu
    var_trace[0] = P
    Q = g**2 + q  # combined process noise covariance
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        u = control_func(t[i - 1])  # control input at time t[i-1]

        Phi = np.exp(a * dt)
        int_u = dt * b * u  # integral of b*u from t[i-1] to t[i]
        mean_trace[i] = Phi * mean_trace[i - 1] + int_u

        # Variance update
        var_trace[i] = (Phi**2) * var_trace[i - 1] + Q * dt
    return mean_trace, var_trace
