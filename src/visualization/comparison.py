import matplotlib.pyplot as plt

from visualization.planning import plot_covariance_ellipse
from visualization.style import figsize as _figsize, save_figure


def plot_two_gap_comparison(
    env,
    det_result,
    pdstl_result,
    *,
    save_png=None,
    save_pdf=None,
    n_ellipses=6,
):
    """
    Plot deterministic and pdSTL planned trajectories with covariance ellipses.
    """
    fig, axes = plt.subplots(1, 2, figsize=_figsize("double", aspect=0.42), sharex=True, sharey=True)

    cases = [
        ("Deterministic nominal baseline", det_result),
        ("pdSTL uncertainty-aware planner", pdstl_result),
    ]

    for ax, (title, result) in zip(axes, cases):
        env.draw_on_ax(ax)

        mean_trace = result["mean_trace"][0].detach().cpu()
        cov_trace = result["cov_trace"][0].detach().cpu()
        T = mean_trace.shape[0] - 1
        step_ell = max(1, T // n_ellipses)
        for t in range(0, T + 1, step_ell):
            plot_covariance_ellipse(
                ax, mean_trace[t, :2].numpy(), cov_trace[t, :2, :2].numpy(),
                facecolor="tab:blue", edgecolor="tab:blue", alpha=0.15, zorder=5,
                label="95% CI" if t == 0 else None,
            )

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
        save_figure(fig, save_png, formats=("png",))

    if save_pdf is not None:
        save_figure(fig, save_pdf, formats=("pdf",))

    plt.show()

    return fig, axes


def plot_two_gap_qstd_sweep(
    env,
    results,
    rollouts_by_level,
    q_std_levels,
    *,
    save_png=None,
    save_pdf=None,
    max_rollouts_to_plot=40,
    n_ellipses=5,
):
    """
    Plot the pdSTL planner's trajectory across several q_std levels, one
    panel per level, with Monte Carlo rollouts and covariance ellipses.
    """
    n = len(q_std_levels)
    fig, axes = plt.subplots(
        1, n, figsize=_figsize("double", aspect=0.42 * 2 / n), sharex=True, sharey=True,
    )
    if n == 1:
        axes = [axes]

    for ax, q_std, result, rollouts in zip(axes, q_std_levels, results, rollouts_by_level):
        env.draw_on_ax(ax)

        n_plot = min(max_rollouts_to_plot, rollouts.shape[0])
        for i in range(n_plot):
            r = rollouts[i].detach().cpu()
            ax.plot(r[:, 0], r[:, 1], linewidth=0.6, alpha=0.25)

        mean_trace = result["mean_trace"][0].detach().cpu()
        cov_trace = result["cov_trace"][0].detach().cpu()
        T = mean_trace.shape[0] - 1
        step_ell = max(1, T // n_ellipses)
        for t in range(0, T + 1, step_ell):
            plot_covariance_ellipse(
                ax, mean_trace[t, :2].numpy(), cov_trace[t, :2, :2].numpy(),
                facecolor="tab:blue", edgecolor="tab:blue", alpha=0.12, zorder=5,
                label="95% CI" if t == 0 else None,
            )

        ax.plot(mean_trace[:, 0], mean_trace[:, 1], linewidth=2.5, label="planned mean")
        ax.scatter(mean_trace[0, 0], mean_trace[0, 1], marker="o", s=60, label="start")
        ax.scatter(mean_trace[-1, 0], mean_trace[-1, 1], marker="x", s=80, label="final")

        ax.set_title(f"$q_\\mathrm{{std}}={q_std}$")
        ax.set_xlabel("x")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("y")

    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(), by_label.keys(), loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, 0.12), frameon=False,
    )

    if save_png is not None:
        save_figure(fig, save_png, formats=("png",))

    if save_pdf is not None:
        save_figure(fig, save_pdf, formats=("pdf",))

    plt.show()

    return fig, axes
