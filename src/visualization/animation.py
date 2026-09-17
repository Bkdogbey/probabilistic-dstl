import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation

from visualization.planning import (
    geometry,
    PALETTE,
    _altitude_axes,
    _draw_altitude_plan,
    _plan,
    _to_np,
    cov_ellipse_params,
    draw_env_on_ax,
)


def animate_results(
    mean_trace,
    cov_trace,
    env,
    filename="trajectory.gif",
    plan_traces=None,
    step=1,
    dt=0.2,
    robot_dims=None,
    title="Motion Planning",
    bounds=None,
):
    """Animate mean trajectory with covariance ellipses; save to file or display."""
    mean_np = mean_trace.detach().cpu().squeeze().numpy()
    cov_np = cov_trace.detach().cpu().squeeze().numpy()

    T = mean_np.shape[0]

    fig, ax = plt.subplots(figsize=(12, 6))

    if bounds:
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
    else:
        ax.set_xlim(-5, 15)
        ax.set_ylim(-4, 8)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    draw_env_on_ax(ax, env, draw_moving_path=False)

    moving_patches = []
    for obs in geometry(env).moving_obstacles:
        w, h = obs["width"], obs["height"]
        xt = np.asarray(
            obs["x_traj"].detach().cpu() if isinstance(obs["x_traj"], torch.Tensor)
            else obs["x_traj"]
        )
        yt = np.asarray(
            obs["y_traj"].detach().cpu() if isinstance(obs["y_traj"], torch.Tensor)
            else obs["y_traj"]
        )
        rect = patches.Rectangle(
            (0, 0), w, h,
            facecolor=PALETTE["obs_moving"]["fill"],
            edgecolor=PALETTE["obs_moving"]["stroke"],
            alpha=0.5,
            label="Moving Obs",
            zorder=6,
        )
        ax.add_patch(rect)
        moving_patches.append((rect, w, h, xt, yt))

    if robot_dims:
        robot_rect = patches.Rectangle(
            (0, 0), robot_dims[0], robot_dims[1],
            facecolor=PALETTE["ego"]["fill"],
            edgecolor=PALETTE["ego"]["stroke"],
            alpha=0.7,
            label="Ego Vehicle",
            zorder=10,
        )
        ax.add_patch(robot_rect)
        robot_dot = None
    else:
        (robot_dot,) = ax.plot(
            [], [], color=PALETTE["ego"]["stroke"], marker="o", markersize=8,
            label="Robot Mean", zorder=10,
        )

    (trail,) = ax.plot([], [], color=PALETTE["ego"]["stroke"], linewidth=2, alpha=0.6, zorder=9)
    (plan_line,) = ax.plot(
        [], [],
        color=PALETTE["plan"]["stroke"],
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Planned Horizon",
    )

    ellipse = patches.Ellipse(
        (0, 0), width=0, height=0, angle=0,
        facecolor=PALETTE["ego"]["fill"],
        edgecolor=PALETTE["ego"]["stroke"],
        alpha=0.25,
        zorder=8,
    )
    ax.add_patch(ellipse)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        if robot_dot:
            robot_dot.set_data([], [])
        if robot_dims:
            robot_rect.set_visible(False)
        trail.set_data([], [])
        plan_line.set_data([], [])
        ellipse.set_width(0)
        ellipse.set_height(0)
        time_text.set_text("")

        actors = [trail, ellipse, time_text, plan_line]
        if robot_dot:
            actors.append(robot_dot)
        if robot_dims:
            actors.append(robot_rect)
        actors.extend([p[0] for p in moving_patches])
        return actors

    def update(frame):  # noqa: C901
        x, y = mean_np[frame, 0], mean_np[frame, 1]

        if bounds is None:
            ax.set_xlim(x - 8.0, x + 12.0)
            ax.set_ylim(-4.0, 8.0)

        if robot_dot:
            robot_dot.set_data([x], [y])

        if robot_dims:
            robot_rect.set_visible(True)
            if frame < len(mean_np) - 1:
                dx = mean_np[frame + 1, 0] - x
                dy = mean_np[frame + 1, 1] - y
            else:
                dx = x - mean_np[frame - 1, 0]
                dy = y - mean_np[frame - 1, 1]

            theta = np.degrees(np.arctan2(dy, dx))
            w, h = robot_dims[0], robot_dims[1]
            t = transforms.Affine2D().translate(-w / 2, -h / 2).rotate_deg(theta).translate(x, y)
            robot_rect.set_transform(t + ax.transData)

        trail.set_data(mean_np[: frame + 1, 0], mean_np[: frame + 1, 1])

        theta, width, height = cov_ellipse_params(cov_np[frame, :2, :2])
        ellipse.set_center((x, y))
        ellipse.set_width(width)
        ellipse.set_height(height)
        ellipse.set_angle(theta)

        time_text.set_text(f"Time Step: {frame}")

        if plan_traces is not None and frame < len(plan_traces):
            plan = plan_traces[frame].detach().cpu().squeeze().numpy()
            plan_line.set_data(plan[:, 0], plan[:, 1])
        else:
            plan_line.set_data([], [])

        for rect, w, h, xt, yt in moving_patches:
            idx = min(frame, len(xt) - 1)
            rect.set_xy((xt[idx] - w / 2, yt[idx] - h / 2))

        actors = [trail, ellipse, time_text, plan_line]
        if robot_dot:
            actors.append(robot_dot)
        if robot_dims:
            actors.append(robot_rect)
        actors.extend([p[0] for p in moving_patches])
        return actors

    frames = range(0, T, step)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=100)

    if filename:
        print(f"Saving animation to {filename}...")
        try:
            if filename.endswith(".gif"):
                ani.save(filename, writer="pillow", fps=20)
            else:
                ani.save(filename, writer="ffmpeg", fps=20)
            print("Animation saved successfully.")
        except Exception as e:
            print(f"Warning: Could not save animation (ffmpeg/pillow issue?): {e}")
            print("Displaying plot instead.")
            plt.show()
    else:
        plt.show()


def animate_altitude_optimization(result, *, dt, threshold, u_max, filename=None, fps=6):
    """One frame per optimizer iterate up to the returned plan, held for one second at the end."""
    returned = result["returned_iteration"]
    frames = result["frames"][: returned + 1]
    H = len(_to_np(frames[0]["controls"]))
    bands = [
        (_to_np(f["mean"])[:, 0], 2 * np.sqrt(_to_np(f["cov"])[:, 0, 0])) for f in frames
    ]
    low = min((mean - band).min() for mean, band in bands)
    high = max((mean + band).max() for mean, band in bands)
    state_ylim = (min(low, threshold) - 0.3, high + 0.3)

    fig, axes = _altitude_axes(dt, H, threshold, u_max, state_ylim)
    static_lines = [len(ax.lines) for ax in axes]  # threshold and control bounds
    initial = _plan(result, "initial")

    def update(k):
        for ax, n_static in zip(axes, static_lines):
            for artist in list(ax.lines[n_static:]) + list(ax.collections):
                artist.remove()
        _draw_altitude_plan(axes, initial, dt=dt, name="Initial", style="--",
                            color=PALETTE["lane"]["stroke"], alpha=0.5)
        _draw_altitude_plan(axes, frames[k], dt=dt, name="Iterate", style="-",
                            color=PALETTE["ego"]["stroke"])
        for ax in axes:
            ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        label = (f"Returned plan (iteration {k + 1})" if k == returned
                 else f"Optimizer iteration {k + 1} / {returned + 1}")
        fig.suptitle(
            f"{label}    R_lower = {frames[k]['interval'][0]:.4f}",
            fontsize=14, fontweight="bold", y=0.985,
        )

    fig.subplots_adjust(left=0.07, right=0.72, top=0.91, bottom=0.07, hspace=0.35)
    sequence = list(range(len(frames))) + [returned] * fps
    ani = FuncAnimation(fig, update, frames=sequence, blit=False)
    if filename:
        ani.save(filename, writer="pillow", fps=fps)
    plt.close(fig)
    return ani
