import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_signal(ax, time, mean_trace, var_trace):
    has_signal = mean_trace is not None and var_trace is not None

    if has_signal:
        mean_trace = np.asarray(mean_trace)
        var_trace = np.asarray(var_trace)
        sigma = np.sqrt(var_trace)
        lower_sigma = mean_trace - sigma
        upper_sigma = mean_trace + sigma

        # Signal + uncertainty
        ax.plot(time, mean_trace, linewidth=2, label="Mean trajectory")
        ax.fill_between(
            time,
            lower_sigma,
            upper_sigma,
            alpha=0.2,
            label="±1σ uncertainty",
        )
        ax.set_title(
            "Signal and uncertainty (input to STL spec)",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)


def plot_robustness(ax, time, robustness_trace):
    ax.plot(time, robustness_trace[:, 0])
    ax.grid()


def plot_stl_formula_bounds(
    time,
    robustness_trace,
    mean_trace=None,
    var_trace=None,
):
    """
    Minimal, generic visualization for ANY STL formula.

    - Accepts output of formula(belief_trajectory).
    - Extracts lower & upper probability.
    - If only a single value is returned, repeats it across time.
    """

    robustness_trace = torch.squeeze(robustness_trace)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    plot_signal(axs[0], time, mean_trace, var_trace)
    plot_robustness(axs[1], time, robustness_trace)

    plt.tight_layout()
    plt.show()
