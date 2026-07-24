"""Shared IEEE-paper plotting style: palette, figure sizing, rcParams, export.

Single source of truth so every script in `visualization`/`planning`/the
crazyflie experiment renders with the same fonts, sizes, and colors instead of
each picking its own figsize/fontsize/dpi.
"""

import os

import matplotlib.pyplot as plt

# Tableau 10 colors, semantic roles shared across all environment/trajectory plots.
PALETTE = {
    "ego": {"fill": "#1f77b4", "stroke": "#1f77b4"},  # Tableau Blue
    "plan": {"fill": "#ff7f0e", "stroke": "#ff7f0e"},  # Tableau Orange
    "visit": {"fill": "#c5b0d5", "stroke": "#9467bd"},  # Tableau Purple (Light/Dark)
    "obs_static": {"fill": "#ff9896", "stroke": "#d62728"},  # Tableau Red (Light/Dark)
    "obs_moving": {
        "fill": "#c49c94",
        "stroke": "#8c564b",
    },  # Tableau Brown (Light/Dark) -- was duplicating obs_static's red
    "lane": {"fill": "#c7c7c7", "stroke": "#7f7f7f"},  # Tableau Gray (Light/Dark)
    "goal": {"fill": "#98df8a", "stroke": "#2ca02c"},  # Tableau Green (Light/Dark)
    "road": {"fill": "#F2F2F7"},  # Light Gray Background
}

IEEE_COL_WIDTH_IN = 3.5  # single-column width
IEEE_TEXT_WIDTH_IN = 7.16  # double-column / full text width

# sqrt(chi2.ppf(0.95, df=2)) -- the correct 2-D 95% confidence-ellipse scale
# factor. 1.96 is the 1-D z-value; using it for a 2-D ellipse under-covers
# (~85% actual coverage, not 95%).
CONFIDENCE_95_K = 2.4477

_MODE_BASE_FONT_SIZE = {"paper": 10, "screen": 14}


def figsize(width="single", aspect=0.75):
    """Return an IEEE-column-appropriate (w, h) figure size in inches.

    width: "single" (3.5in), "double" (7.16in), or an explicit width in inches.
    aspect: height / width ratio.
    """
    if width == "single":
        w = IEEE_COL_WIDTH_IN
    elif width == "double":
        w = IEEE_TEXT_WIDTH_IN
    else:
        w = float(width)
    return (w, w * aspect)


def _rc_for_mode(mode):
    base = _MODE_BASE_FONT_SIZE.get(mode, _MODE_BASE_FONT_SIZE["paper"])
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
        "figure.titlesize": base + 2,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Embed TrueType (42) instead of Type-3 fonts -- avoids IEEE PDF eXpress
        # font-embedding warnings on camera-ready submission.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def set_ieee_style(mode="paper"):
    """Set matplotlib rcParams globally for IEEE-style figures.

    mode="paper": compact fonts/figures sized for camera-ready inclusion.
    mode="screen": larger fonts, for interactive/debugging use.
    """
    plt.rcParams.update(_rc_for_mode(mode))


def screen_context():
    """Context manager applying "screen" (larger, interactive-friendly) sizing.

    For figures meant for on-screen viewing or GIF export (animations, live
    MPC plots) so they read clearly regardless of the global paper/screen mode.
    """
    return plt.rc_context(_rc_for_mode("screen"))


def save_figure(fig, path, formats=("pdf", "png"), dpi=300):
    """Save `fig` to `path` (extension stripped/ignored) in each of `formats`.

    Always includes a vector PDF (for \\includegraphics in the paper) plus a
    raster PNG preview by default. Returns {format: written_path}.
    """
    stem, _ = os.path.splitext(str(path))
    out_dir = os.path.dirname(stem)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    written = {}
    for fmt in formats:
        out_path = f"{stem}.{fmt}"
        save_kwargs = {"bbox_inches": "tight"}
        if fmt != "pdf":
            save_kwargs["dpi"] = dpi
        fig.savefig(out_path, **save_kwargs)
        written[fmt] = out_path
    return written
