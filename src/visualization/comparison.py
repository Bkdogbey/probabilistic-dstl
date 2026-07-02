import os

import matplotlib.pyplot as plt


def plot_two_gap_comparison(
    env,
    det_result,
    pdstl_result,
    det_rollouts,
    pdstl_rollouts,
    *,
    save_png=None,
    save_pdf=None,
    max_rollouts_to_plot=40,
):
    """
    Plot deterministic and pdSTL trajectories with Monte Carlo rollouts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

    cases = [
        ("Deterministic nominal baseline", det_result, det_rollouts),
        ("pdSTL uncertainty-aware planner", pdstl_result, pdstl_rollouts),
    ]

    for ax, (title, result, rollouts) in zip(axes, cases):
        env.draw_on_ax(ax)

        n_plot = min(max_rollouts_to_plot, rollouts.shape[0])

        for i in range(n_plot):
            r = rollouts[i].detach().cpu()
            ax.plot(r[:, 0], r[:, 1], linewidth=0.6, alpha=0.25)

        mean_trace = result["mean_trace"][0].detach().cpu()
        ax.plot(mean_trace[:, 0], mean_trace[:, 1], linewidth=2.5, label="planned mean")

        ax.scatter(mean_trace[0, 0], mean_trace[0, 1], marker="o", s=60, label="start")
        ax.scatter(mean_trace[-1, 0], mean_trace[-1, 1], marker="x", s=80, label="final")

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()

    if save_png is not None:
        os.makedirs(os.path.dirname(save_png), exist_ok=True)
        fig.savefig(save_png, dpi=300, bbox_inches="tight")

    if save_pdf is not None:
        os.makedirs(os.path.dirname(save_pdf), exist_ok=True)
        fig.savefig(save_pdf, bbox_inches="tight")

    plt.show()

    return fig, axes
