"""Gaussian altitude demonstrations for the offline temporal operators."""

from models.dynamics import always_altitude_example, eventually_altitude_example
from pdstl.operators import Always, Eventually, GreaterThan
from utils import create_belief_trajectory
from visualization.temporal import plot_temporal_example


def _print_results(time, mean, variance, atomic, temporal, interval, label):
    print(f"\n{label}")
    print("t       mean      std       p_lower   p_upper")
    for t, m, v, (lower, upper) in zip(time, mean, variance, atomic[0]):
        print(f"{t:.4f}  {m:.4f}   {v.sqrt():.4f}    {lower:.4f}    {upper:.4f}")
    print("origin  window_start  window_end  lower     upper")
    a, b = interval
    for k, (lower, upper) in enumerate(temporal[0]):
        print(
            f"{time[k]:.4f}  {time[k + a]:.4f}        {time[k + b]:.4f}      "
            f"{lower:.4f}    {upper:.4f}"
        )
    print("Temporal values are stochastic robustness, not trajectory probabilities.")


def _run_temporal(trace, operator, threshold, interval, show):
    time, mean, variance = trace
    beliefs = create_belief_trajectory(
        mean, variance, dtype=mean.dtype, device=mean.device
    )
    predicate = GreaterThan(threshold)
    formula = operator(predicate, interval=interval)
    atomic = predicate(beliefs)
    temporal = formula(beliefs, scale=-1)
    label = (
        f"{operator.__name__}[{interval[0]},{interval[1]}](altitude >= {threshold:g} m)"
    )
    _print_results(time, mean, variance, atomic, temporal, interval, label)
    figure = plot_temporal_example(
        time, mean, variance, atomic, temporal, threshold, label, show=show
    )
    return time, mean, variance, atomic, temporal, figure


def run_always_example(threshold=50.0, interval=(0, 1), show=True):
    """Evaluate the altitude dip with Always and print every complete window."""
    return _run_temporal(always_altitude_example(), Always, threshold, interval, show)


def run_eventually_example(threshold=55.0, interval=(0, 1), show=True):
    """Evaluate the altitude climb with Eventually and print every complete window."""
    return _run_temporal(
        eventually_altitude_example(), Eventually, threshold, interval, show
    )
