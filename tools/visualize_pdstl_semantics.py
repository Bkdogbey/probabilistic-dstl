#!/usr/bin/env python3
"""Generate deterministic validation figures for pdSTL semantics Patches 1–4.

Legacy equations live only in explicitly named visualization helpers. Current
quantities are evaluated through production operators. Every plot performs its
mathematical assertions before it is written.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pdstl-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import multivariate_normal, norm

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pdstl.base import BeliefTrajectory  # noqa: E402
from pdstl.operators import And, Maxish, Minish, STL_Formula, Until  # noqa: E402
from pdstl.probability import gaussian_residual_probability  # noqa: E402
from planning.environment import RectangularGoalPredicate  # noqa: E402
from planning.planner import TorchGaussianBelief  # noqa: E402
from visualization.style import PALETTE, figsize, save_figure, set_ieee_style  # noqa: E402

logging.getLogger("fontTools.subset").setLevel(logging.WARNING)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures" / "semantic_validation"
BETAS = np.array([1, 2, 5, 10, 20, 50, 100, 500], dtype=float)
UNTIL_PHI = np.array(
    [[0.9, 0.9], [0.1, 0.3], [0.8, 0.9], [0.5, 0.7]], dtype=float
)
UNTIL_PSI = np.array(
    [[0.2, 0.3], [0.9, 0.95], [0.4, 0.6], [0.8, 0.9]], dtype=float
)
UNTIL_INTERVAL = [0, 2]


class _TensorFormula(STL_Formula):
    """Test/visualization formula returning an existing tensor unchanged."""

    def __init__(self, trace):
        super().__init__()
        self.trace = trace

    def robustness_trace(self, belief_trajectory, **kwargs):
        return self.trace


def _save(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = save_figure(fig, output_dir / stem, formats=("pdf", "png"))
    plt.close(fig)
    for path in written.values():
        assert Path(path).is_file() and Path(path).stat().st_size > 0
    return [Path(written["pdf"]), Path(written["png"])]


def _legacy_atomic_for_visualization(mean, confidence_shift=2.0):
    """Reconstruct the former confidence-shifted unit-variance atom."""
    mean = np.asarray(mean, dtype=float)
    return norm.cdf(mean - confidence_shift), norm.cdf(mean + confidence_shift)


def _legacy_rectangle_for_visualization(marginal_probabilities):
    """Reconstruct the former independence-product generic box value."""
    return np.prod(np.asarray(marginal_probabilities, dtype=float), axis=0)


def _direct_exact_until(phi, psi, interval):
    """Direct bounded inclusive-prefix StoRI Until reference."""
    phi = np.asarray(phi, dtype=float)
    psi = np.asarray(psi, dtype=float)
    horizon = phi.shape[0]
    a, b = map(int, interval)
    out = np.zeros_like(phi)
    for t in range(horizon):
        candidates = []
        for tau in range(t + a, min(t + b, horizon - 1) + 1):
            prefix = np.min(phi[t : tau + 1], axis=0)
            candidates.append(
                [
                    max(prefix[0] + psi[tau, 0] - 1.0, 0.0),
                    min(prefix[1], psi[tau, 1]),
                ]
            )
        if candidates:
            out[t] = np.max(candidates, axis=0)
    return out


def _legacy_until_for_visualization(phi, psi, interval):
    """Reconstruct the former exclusive-prefix, pointwise-min Until."""
    phi = np.asarray(phi, dtype=float)
    psi = np.asarray(psi, dtype=float)
    horizon = phi.shape[0]
    a, b = map(int, interval)
    out = np.zeros_like(phi)
    for t in range(horizon):
        candidates = []
        for tau in range(t + a, min(t + b, horizon - 1) + 1):
            prefix = np.ones(2) if tau == t else np.min(phi[t:tau], axis=0)
            candidates.append(np.minimum(prefix, psi[tau]))
        if candidates:
            out[t] = np.max(candidates, axis=0)
    return out


def _production_until(phi, psi, interval, scale=-1):
    phi_tensor = torch.as_tensor(phi, dtype=torch.float64).unsqueeze(0)
    psi_tensor = torch.as_tensor(psi, dtype=torch.float64).unsqueeze(0)
    result = Until(
        _TensorFormula(phi_tensor), _TensorFormula(psi_tensor), interval=interval
    )(None, scale=scale)
    return result.detach().cpu().numpy()[0]


def plot_atomic_probability(output_dir):
    means = np.linspace(-4.0, 4.0, 401)
    means_tensor = torch.as_tensor(means, dtype=torch.float64)
    corrected = gaussian_residual_probability(
        means_tensor, torch.ones_like(means_tensor)
    ).numpy()
    former_lower, former_upper = _legacy_atomic_for_visualization(means)

    equality = gaussian_residual_probability(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    ).item()
    deterministic = gaussian_residual_probability(
        means_tensor, torch.zeros_like(means_tensor)
    ).numpy()
    assert equality == 0.5
    assert np.isclose(former_lower[200], norm.cdf(-2.0), atol=1e-12)
    assert np.isclose(former_upper[200], norm.cdf(2.0), atol=1e-12)
    assert deterministic[200] == 1.0

    fig, axes = plt.subplots(1, 2, figsize=figsize("double", aspect=0.38))
    ax = axes[0]
    ax.fill_between(
        means,
        former_lower,
        former_upper,
        color="#bdbdbd",
        alpha=0.45,
        label="Former shifted interval ($k=2$)",
    )
    ax.plot(means, former_lower, color="#7f7f7f", linestyle="--", linewidth=1.0)
    ax.plot(means, former_upper, color="#7f7f7f", linestyle="--", linewidth=1.0)
    ax.plot(
        means,
        corrected,
        color=PALETTE["ego"]["stroke"],
        label=r"Correct/current $\Phi(m)$",
    )
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.scatter([0.0], [0.5], color=PALETTE["ego"]["stroke"], zorder=4)
    ax.annotate(
        "correct: 0.5\nformer: [0.02275, 0.97725]",
        xy=(0.0, 0.5),
        xytext=(0.55, 0.38),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    ax.set(xlabel="Residual mean $m$", ylabel=r"$P(R\geq 0)$", ylim=(-0.03, 1.03))
    ax.set_title("Unit-variance Gaussian atom")
    ax.grid(True)
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.step(means, deterministic, where="post", color=PALETTE["goal"]["stroke"])
    ax.scatter([0.0], [1.0], color=PALETTE["goal"]["stroke"], zorder=4)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set(xlabel="Residual mean $m$", ylabel=r"$P(R\geq 0)$", ylim=(-0.03, 1.03))
    ax.set_title(r"Zero variance: $\mathbf{1}\{m\geq0\}$")
    ax.grid(True)
    fig.tight_layout()
    return _save(fig, output_dir, "01_atomic_probability")


def _generic_rectangle_bounds(rho):
    covariance = torch.tensor(
        [[[1.0, rho], [rho, 1.0]]], dtype=torch.float64
    )
    belief = TorchGaussianBelief(torch.zeros(1, 2, dtype=torch.float64), covariance)
    trajectory = BeliefTrajectory([belief])
    result = RectangularGoalPredicate({"x": [-1.0, 1.0], "y": [-1.0, 1.0]})(
        trajectory
    )
    return result[0, 0].detach().cpu().numpy()


def plot_rectangle_frechet(output_dir):
    rhos = np.linspace(-0.95, 0.95, 191)
    rng = np.random.default_rng(20260814)
    true_inside = np.array(
        [
            multivariate_normal.cdf(
                [1.0, 1.0],
                mean=[0.0, 0.0],
                cov=[[1.0, rho], [rho, 1.0]],
                lower_limit=[-1.0, -1.0],
                rng=rng,
            )
            for rho in rhos
        ]
    )
    production_bounds = np.array([_generic_rectangle_bounds(rho) for rho in rhos])
    lower, upper = production_bounds[:, 0], production_bounds[:, 1]
    axis_probability = norm.cdf(1.0) - norm.cdf(-1.0)
    legacy_product = _legacy_rectangle_for_visualization(
        [axis_probability, axis_probability]
    )

    expected_bounds = np.array(
        [max(0.0, 2.0 * axis_probability - 1.0), axis_probability]
    )
    np.testing.assert_allclose(
        production_bounds,
        np.broadcast_to(expected_bounds, production_bounds.shape),
        atol=1e-12,
    )
    assert np.all(lower - 1e-10 <= true_inside)
    assert np.all(true_inside <= upper + 1e-10)
    assert abs(true_inside[np.argmin(abs(rhos))] - legacy_product) < 1e-8
    assert np.max(np.abs(true_inside - legacy_product)) > 0.1

    fig, axes = plt.subplots(1, 2, figsize=figsize("double", aspect=0.4))
    ax = axes[0]
    ax.fill_between(rhos, lower, upper, color="#c6dbef", label="Generic Fréchet interval")
    ax.plot(rhos, true_inside, color=PALETTE["ego"]["stroke"], label="True Gaussian box probability")
    ax.plot(rhos, np.full_like(rhos, legacy_product), "--", color=PALETTE["obs_static"]["stroke"], label="Former independence product")
    ax.plot(rhos, lower, color="#6baed6", linewidth=0.8)
    ax.plot(rhos, upper, color="#6baed6", linewidth=0.8)
    ax.set(xlabel=r"Correlation $\rho$", ylabel="Inside-box probability", ylim=(0.25, 0.75))
    ax.set_title(r"Containment in $[-1,1]^2$")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    true_outside = 1.0 - true_inside
    safe_lower, safe_upper = 1.0 - upper, 1.0 - lower
    assert np.all(safe_lower - 1e-10 <= true_outside)
    assert np.all(true_outside <= safe_upper + 1e-10)
    ax = axes[1]
    ax.fill_between(rhos, safe_lower, safe_upper, color="#fee0d2", label="Obstacle-complement interval")
    ax.plot(rhos, true_outside, color=PALETTE["obs_static"]["stroke"], label="True outside probability")
    ax.plot(rhos, np.full_like(rhos, 1.0 - legacy_product), "--", color=PALETTE["ego"]["stroke"], label="Former independence complement")
    ax.plot(rhos, safe_lower, color="#fc9272", linewidth=0.8)
    ax.plot(rhos, safe_upper, color="#fc9272", linewidth=0.8)
    ax.set(xlabel=r"Correlation $\rho$", ylabel="Outside-box probability", ylim=(0.25, 0.75))
    ax.set_title("Obstacle complement")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _save(fig, output_dir, "02_rectangle_frechet")


def plot_until_correction(output_dir):
    corrected = _production_until(UNTIL_PHI, UNTIL_PSI, UNTIL_INTERVAL)
    reference = _direct_exact_until(UNTIL_PHI, UNTIL_PSI, UNTIL_INTERVAL)
    former = _legacy_until_for_visualization(UNTIL_PHI, UNTIL_PSI, UNTIL_INTERVAL)
    np.testing.assert_allclose(corrected, reference, atol=1e-12)
    assert np.all((0.0 <= corrected[:, 0]) & (corrected[:, 0] <= corrected[:, 1]))
    assert np.all(corrected[:, 1] <= 1.0)
    assert np.max(np.abs(former - corrected)) >= 0.5

    times = np.arange(len(corrected))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize("single", aspect=1.25))
    labels = ["Lower endpoint", "Upper endpoint"]
    for endpoint, ax in enumerate(axes):
        ax.plot(times, corrected[:, endpoint], "o-", color=PALETTE["ego"]["stroke"], label="Correct exact StoRI")
        ax.plot(times, former[:, endpoint], "s--", color=PALETTE["obs_static"]["stroke"], label="Former recursion")
        ax.set_ylabel(labels[endpoint])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True)
    difference = np.max(np.abs(former - corrected), axis=1)
    marked_time = int(np.argmax(difference))
    axes[0].annotate(
        f"inclusive/Fréchet correction\nΔ={difference[marked_time]:.2f}",
        xy=(marked_time, corrected[marked_time, 0]),
        xytext=(0.15, 0.48),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    axes[0].legend(loc="upper right")
    axes[0].set_title(r"Bounded exact Until, $I=[0,2]$")
    axes[1].set_xlabel("Evaluation time $t$")
    axes[1].set_xticks(times)
    fig.tight_layout()
    return _save(fig, output_dir, "03_until_correction")


def plot_audit_progress(output_dir):
    stages = ["Initial\naudit", "Patch 1", "Patch 2", "Patch 3", "Patch 4"]
    remaining = np.array([30, 23, 22, 20, 15])
    assert np.all(np.diff(remaining) <= 0)
    fig, ax = plt.subplots(figsize=figsize("single", aspect=0.72))
    bars = ax.bar(stages, remaining, color=["#7f7f7f", "#9ecae1", "#6baed6", "#3182bd", "#08519c"])
    ax.bar_label(bars, padding=2)
    ax.set_ylabel("Remaining known audit findings")
    ax.set_ylim(0, 34)
    ax.set_title("Semantic-audit progress")
    ax.grid(True, axis="y")
    fig.tight_layout()
    return _save(fig, output_dir, "04_audit_progress")


def _current_reduction(values, beta, operation):
    tensor = torch.as_tensor(values, dtype=torch.float64).reshape(1, -1, 1)
    reducer = Minish() if operation == "min" else Maxish()
    return reducer(tensor, beta, dim=1, keepdim=False).item()


def plot_smoothing_extrema(output_dir):
    examples = [
        ("low constant", np.array([0.05, 0.05, 0.05])),
        ("nonconstant", np.array([0.20, 0.35, 0.80])),
        ("high constant", np.array([0.99, 0.99, 0.99])),
    ]
    fig, axes = plt.subplots(2, 3, figsize=figsize("double", aspect=0.7), sharex="col")
    for column, (name, values) in enumerate(examples):
        smooth_min = np.array([_current_reduction(values, beta, "min") for beta in BETAS])
        smooth_max = np.array([_current_reduction(values, beta, "max") for beta in BETAS])
        exact_min, exact_max = values.min(), values.max()
        min_error = np.abs(smooth_min - exact_min)
        max_error = np.abs(smooth_max - exact_max)

        ax = axes[0, column]
        ax.plot(BETAS, smooth_min, "o-", label="Current smooth min", color=PALETTE["ego"]["stroke"])
        ax.plot(BETAS, smooth_max, "s-", label="Current smooth max", color=PALETTE["obs_static"]["stroke"])
        ax.axhline(exact_min, color=PALETTE["ego"]["stroke"], linestyle=":", label="Exact min")
        ax.axhline(exact_max, color=PALETTE["obs_static"]["stroke"], linestyle=":", label="Exact max")
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.axhline(1.0, color="black", linewidth=0.7)
        ax.set_xscale("log")
        ax.set_ylim(-1.1, 2.1)
        ax.set_title(f"{name}: {values.tolist()}")
        ax.grid(True)
        if column == 0:
            ax.set_ylabel("Reduction value")

        ax = axes[1, column]
        ax.plot(BETAS, min_error, "o-", label="Min error", color=PALETTE["ego"]["stroke"])
        ax.plot(BETAS, max_error, "s-", label="Max error", color=PALETTE["obs_static"]["stroke"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Smoothing scale $\beta$")
        ax.grid(True)
        if column == 0:
            ax.set_ylabel("Absolute error")

    assert _current_reduction([0.05] * 3, 10, "min") < 0.0
    assert _current_reduction([0.99] * 3, 10, "max") > 1.0
    axes[0, 0].legend(loc="upper right", fontsize=7)
    axes[1, 0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return _save(fig, output_dir, "05_smoothing_extrema")


def _smooth_min_gradient(values, beta):
    tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True).reshape(1, -1, 1)
    result = Minish()(tensor, beta, dim=1, keepdim=False)
    gradient = torch.autograd.grad(result.sum(), tensor)[0]
    return gradient.detach().cpu().numpy().reshape(-1)


def plot_smoothing_gradients(output_dir):
    values = np.array([0.20, 0.21, 0.60, 0.90])
    betas = [1, 5, 10, 50, 100]
    gradients = np.array([_smooth_min_gradient(values, beta) for beta in betas])
    hard_gradient = _smooth_min_gradient(values, -1)
    np.testing.assert_allclose(gradients.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(hard_gradient, [1.0, 0.0, 0.0, 0.0], atol=0.0)
    assert gradients[0, 1] > 0.2
    assert gradients[-1, 0] > gradients[0, 0]

    fig, ax = plt.subplots(figsize=figsize("single", aspect=0.78))
    for beta, gradient in zip(betas, gradients):
        ax.plot(np.arange(len(values)), gradient, "o-", label=rf"$\beta={beta}$")
    ax.plot(np.arange(len(values)), hard_gradient, "k--", marker="x", label="Exact hard min")
    ax.set_xticks(np.arange(len(values)), [f"$x_{i}$\n{value:.2f}" for i, value in enumerate(values)])
    ax.set(xlabel="Temporal element", ylabel="Gradient mass", ylim=(-0.03, 1.03))
    ax.set_title("Current smooth-min gradient distribution")
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return _save(fig, output_dir, "06_smoothing_gradients")


def _and_surface(scale):
    axis = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    l1_grid, l2_grid = torch.meshgrid(axis, axis, indexing="xy")
    l1 = l1_grid.reshape(-1).clone().requires_grad_(True)
    l2 = l2_grid.reshape(-1).clone().requires_grad_(True)
    ones = torch.ones_like(l1)
    left = torch.stack([l1, ones], dim=-1).unsqueeze(0)
    right = torch.stack([l2, ones], dim=-1).unsqueeze(0)
    output = And(_TensorFormula(left), _TensorFormula(right))(None, scale=scale)[0, :, 0]
    grad_l1, grad_l2 = torch.autograd.grad(output.sum(), (l1, l2))
    return (
        l1_grid.numpy(),
        l2_grid.numpy(),
        output.detach().numpy().reshape(101, 101),
        torch.sqrt(grad_l1.square() + grad_l2.square()).detach().numpy().reshape(101, 101),
    )


def plot_and_gradient_region(output_dir):
    l1, l2, exact_output, exact_gradient = _and_surface(-1)
    _, _, smooth_output, smooth_gradient = _and_surface(50)
    np.testing.assert_array_equal(exact_output, smooth_output)
    np.testing.assert_array_equal(exact_gradient, smooth_gradient)
    assert np.all(exact_output[l1 + l2 < 1.0] == 0.0)
    assert np.all(exact_gradient[l1 + l2 < 1.0] == 0.0)

    fig, axes = plt.subplots(1, 2, figsize=figsize("double", aspect=0.42))
    levels = np.linspace(0.0, 1.0, 11)
    contour = axes[0].contourf(l1, l2, exact_output, levels=levels, cmap="Blues")
    fig.colorbar(contour, ax=axes[0], label=r"$r^\downarrow_{\wedge}$")
    gradient_contour = axes[1].contourf(l1, l2, exact_gradient, levels=10, cmap="Oranges")
    fig.colorbar(gradient_contour, ax=axes[1], label="Gradient magnitude")
    for ax in axes:
        ax.plot([0.0, 1.0], [1.0, 0.0], "k--", linewidth=1.0)
        ax.fill_between([0.0, 1.0], [1.0, 0.0], [0.0, 0.0], color="white", alpha=0.18, hatch="///")
        ax.set(xlabel=r"$l_1$", ylabel=r"$l_2$", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_title(r"$\max(l_1+l_2-1,0)$")
    axes[1].set_title("Zero-gradient region below boundary")
    axes[1].text(0.18, 0.20, r"$l_1+l_2<1$", ha="center")
    fig.tight_layout()
    return _save(fig, output_dir, "07_and_gradient_region")


def plot_smooth_until_mismatch(output_dir):
    exact = _production_until(UNTIL_PHI, UNTIL_PSI, UNTIL_INTERVAL, scale=-1)
    reference = _direct_exact_until(UNTIL_PHI, UNTIL_PSI, UNTIL_INTERVAL)
    np.testing.assert_allclose(exact, reference, atol=1e-12)
    errors = np.array(
        [
            np.max(
                np.abs(
                    _production_until(UNTIL_PHI, UNTIL_PSI, UNTIL_INTERVAL, scale=beta)
                    - exact
                )
            )
            for beta in BETAS
        ]
    )
    assert errors[-1] > 0.01

    fig, ax = plt.subplots(figsize=figsize("single", aspect=0.72))
    ax.plot(BETAS, errors, "o-", color=PALETTE["obs_static"]["stroke"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"Smoothing scale $\beta$")
    ax.set_ylabel(r"$\|r^U_\beta-r^U_{\mathrm{exact}}\|_\infty$")
    ax.set_title("Smooth Until mismatch after exact correction")
    ax.grid(True)
    ax.annotate(
        f"β=500: {errors[-1]:.3f}",
        xy=(BETAS[-1], errors[-1]),
        xytext=(35, 0.67),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    fig.tight_layout()
    return _save(fig, output_dir, "08_smooth_until_mismatch")


PLOTTERS = {
    "atomic": plot_atomic_probability,
    "rectangle": plot_rectangle_frechet,
    "until": plot_until_correction,
    "audit": plot_audit_progress,
    "extrema": plot_smoothing_extrema,
    "gradients": plot_smoothing_gradients,
    "and": plot_and_gradient_region,
    "smooth-until": plot_smooth_until_mismatch,
}


def _selected_plotters(selection):
    if selection == "all":
        return list(PLOTTERS.items())
    if selection == "smoothing":
        names = ["extrema", "gradients", "and", "smooth-until"]
        return [(name, PLOTTERS[name]) for name in names]
    return [(selection, PLOTTERS[selection])]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=["all", "smoothing", *PLOTTERS],
        default="all",
        help="Generate all figures, the smoothing group, or one named figure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PDF and PNG outputs.",
    )
    args = parser.parse_args(argv)

    set_ieee_style("paper")
    written = []
    for name, plotter in _selected_plotters(args.figure):
        paths = plotter(args.output_dir)
        written.extend(paths)
        print(f"{name}: " + ", ".join(str(path) for path in paths))
    return written


if __name__ == "__main__":
    main()
