"""Figure for the risk-aversion experiment (experiments/risk_aversion.py).

Pure plotting: the caller supplies the planned belief paths and the Monte
Carlo success rates; nothing here loads results or runs a planner.

Styled like the lane paper figure (visualization.figures.plot_lane_paper):
7 pt DejaVu Sans, obstacles in the light-fill / dark-edge red of the ramp
end, the goal in the light green of the target lane, pdSTL in blue with its
belief, STLCG as the solid orange plan.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Rectangle

from visualization.figures import COLORS, _ellipse_parameters

LABELS = {"stlcg": "STLCG", "pdstl": "pdSTL"}

PAPER_WIDTH = 7.16  # in, IEEE two-column text width
PAPER_FONT = 7  # pt
STYLE = {
    "mathtext.fontset": "dejavusans",
    "font.family": "DejaVu Sans",
    "font.size": PAPER_FONT,
    # Embed TrueType, not Type 3, fonts (IEEE PDF checks reject Type 3).
    "pdf.fonttype": 42,
}
# The lane figure's region colours.
OBSTACLE_FILL, OBSTACLE_EDGE = "#ff9896", "#d62728"
GOAL_FILL, GOAL_EDGE = "#98df8a", "#2ca02c"
WALL = "0.2"
METHOD_STYLE = {
    "pdstl": {"color": COLORS["mean"], "ls": "-"},
    "stlcg": {"color": COLORS["planned"], "ls": "-"},
}
CONFIDENCE = 0.95

# Layout, in inches.
LEFT, RIGHT, GAP = 0.30, 0.08, 0.07  # y labels, right margin, between panels
BOTTOM, TITLE, LEGEND = 0.27, 0.13, 0.15  # x labels, panel title, legend row


def _draw_room(ax, cfg, ylim):
    """Walls, obstacles and goal; start marked S."""
    (x0, x1), (y0, y1) = cfg["bounds"]["x_range"], cfg["bounds"]["y_range"]
    # The walls are the axes, drawn unclipped over the hidden spines and
    # only inside the view `ylim`, so a cut room stays open where it is cut.
    low, high = max(y0, ylim[0]), min(y1, ylim[1])
    for x in (x0, x1):
        ax.plot(
            [x, x],
            [low, high],
            color=WALL,
            lw=1.1,
            zorder=2,
            clip_on=False,
            solid_capstyle="butt",
        )
    for y in (y0, y1):
        if ylim[0] <= y <= ylim[1]:
            ax.plot(
                [x0, x1],
                [y, y],
                color=WALL,
                lw=1.1,
                zorder=2,
                clip_on=False,
                solid_capstyle="projecting",
            )
    for block in cfg["obstacles"]:
        (bx0, bx1), (by0, by1) = block["x_range"], block["y_range"]
        ax.add_patch(
            Rectangle(
                (bx0, by0),
                bx1 - bx0,
                by1 - by0,
                fc=OBSTACLE_FILL,
                ec=OBSTACLE_EDGE,
                hatch="//",
                alpha=0.9,
                lw=0.8,
            )
        )
    (gx0, gx1), (gy0, gy1) = cfg["goal"]["x_range"], cfg["goal"]["y_range"]
    ax.add_patch(
        Rectangle(
            (gx0, gy0),
            gx1 - gx0,
            gy1 - gy0,
            fc=GOAL_FILL,
            ec=GOAL_EDGE,
            alpha=0.6,
            lw=0.8,
        )
    )
    ax.text(
        (gx0 + gx1) / 2,
        gy1 + 0.12,
        "G",
        color=GOAL_EDGE,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    sx, sy = cfg["x0_mean"][:2]
    ax.plot(sx, sy, "o", color="k", ms=3, zorder=5)
    ax.text(sx, sy + 0.2, "S", fontweight="bold", ha="center", va="bottom")


def _draw_plan(ax, mean, cov, method, every):
    """Mean path; pdSTL also gets its 95% belief ellipses every `every`.

    STLCG plans the noise-free mean, so it has no belief to draw; its
    executed spread shows in its Monte Carlo success instead.
    """
    style = METHOD_STYLE[method]
    ax.plot(*mean.T, lw=1.4, zorder=4, **style)
    if method != "pdstl":
        return
    for k in [*range(every, len(mean) - 1, every), len(mean) - 1]:
        width, height, angle = _ellipse_parameters(cov[k], CONFIDENCE)
        ax.add_patch(
            Ellipse(
                mean[k],
                width,
                height,
                angle=angle,
                fc=style["color"],
                ec="none",
                alpha=0.18,
                zorder=3,
            )
        )


def _top(cfg, beliefs, margin=0.45):
    """Upper y limit: the highest drawn mark plus `margin`, inside the room.

    The room above the plans is empty, so it is cut to save space.
    """
    radius = np.sqrt(-2 * np.log(1 - CONFIDENCE))
    highest = max(
        float(np.max(mean[:, 1] + radius * np.sqrt(cov[:, 1, 1])))
        for mean, cov in beliefs.values()
    )
    goal = cfg["goal"]["y_range"][1] + 0.5  # the G label
    room = cfg["bounds"]["y_range"][1]
    return min(room, max(highest, goal) + margin)


def _style_axes(ax, xlim, ylim, first, last):
    """Ticks every 2 m; only the last panel labels the right wall, so the
    labels of neighbouring panels do not meet."""
    ax.set(xlim=xlim, ylim=ylim, aspect="equal")
    xticks = np.arange(xlim[0], xlim[1] + 1e-9, 2)
    ax.set_xticks(xticks if last else xticks[xticks < xlim[1]])
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 1e-9, 2))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(labelsize=PAPER_FONT, length=2, pad=1, color="0.4")
    ax.set_xlabel("x (m)", fontsize=PAPER_FONT, labelpad=0)
    if first:
        ax.set_ylabel("y (m)", fontsize=PAPER_FONT, labelpad=1)
    else:
        ax.tick_params(labelleft=False)


def _region_handles():
    return [
        Patch(
            fc=OBSTACLE_FILL,
            ec=OBSTACLE_EDGE,
            hatch="//",
            alpha=0.9,
            lw=0.8,
            label="Obstacle",
        ),
        Patch(fc=GOAL_FILL, ec=GOAL_EDGE, alpha=0.6, lw=0.8, label="Goal"),
    ]


def plot_risk_aversion(
    cfg,
    beliefs,
    out_dir,
    methods=("stlcg", "pdstl"),
    width=PAPER_WIDTH,
    stem="figure",
):
    """One panel per noise level, both plans in each (two-column width).

    `beliefs[(method, noise)]` is (mean [T, 2], cov [T, 2, 2]). Writes
    `<stem>.png` and `<stem>.pdf` to `out_dir` and returns the png path.

    `width` is in inches: the IEEE two-column width by default, less to
    leave room beside the figure. Fonts stay 7 pt; the panels shrink. The
    panels share one metres-per-inch scale and the empty room above the
    plans is cut (see `_top`).
    """
    levels = list(cfg["noise"])
    every = cfg["figure"]["ellipse_every"]
    xlim = tuple(cfg["bounds"]["x_range"])
    ylim = (cfg["bounds"]["y_range"][0], _top(cfg, beliefs))
    panel_w = (width - LEFT - RIGHT - GAP * (len(levels) - 1)) / len(levels)
    panel_h = panel_w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    height = BOTTOM + panel_h + TITLE + LEGEND
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(width, height))
        for i, noise in enumerate(levels):
            left = LEFT + i * (panel_w + GAP)
            ax = fig.add_axes(
                [
                    left / width,
                    BOTTOM / height,
                    panel_w / width,
                    panel_h / height,
                ]
            )
            _draw_room(ax, cfg, ylim)
            for method in methods:
                mean, cov = beliefs[(method, noise)]
                _draw_plan(ax, mean, cov, method, every)
            _style_axes(ax, xlim, ylim, i == 0, i == len(levels) - 1)
            ax.set_title(
                f"({'abc'[i]}) {noise.capitalize()} noise, "
                f"$q = {cfg['noise'][noise]:.4f}$",
                loc="left",
                fontsize=PAPER_FONT,
                pad=1.5,
            )
        handles = [
            Line2D([], [], lw=1.4, label="pdSTL", **METHOD_STYLE["pdstl"]),
            Patch(
                fc=METHOD_STYLE["pdstl"]["color"],
                alpha=0.25,
                label="95% CI",
            ),
            Line2D([], [], lw=1.4, label="STLCG", **METHOD_STYLE["stlcg"]),
            *_region_handles(),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(handles),
            fontsize=PAPER_FONT - 1,
            frameon=False,
            handlelength=1.4,
            columnspacing=1.0,
            borderaxespad=0.0,
            borderpad=0.1,
        )
        fig.savefig(out_dir / f"{stem}.pdf")
        fig.savefig(out_dir / f"{stem}.png", dpi=300)
        plt.close(fig)
    return out_dir / f"{stem}.png"
