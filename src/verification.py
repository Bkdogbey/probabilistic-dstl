"""Verification suite for the probability-first pdSTL pipeline.

Three demonstrations, driven from ``src/main.py``:

A. ``F[5,10](x >= 8)``       -- temporal **union**.
B. ``G[2,6](3 < z < 5)``     -- probabilistic **conjunction** + temporal
                                **intersection**.
C. ``G Safe AND F Goal``     -- the complete pipeline on a linear stochastic
                                system, plus direct optimization of the hard
                                lower probability bound.

These are *verification* examples -- they exist to show that the implemented
semantics behaves as intended -- not the final application experiments.

Every number reported here comes from the exact hard probability semantics
already in :mod:`pdstl`. Nothing in this module defines, approximates, or
smooths a bound: it builds scenarios, hands atomic event probabilities to
pdSTL, and prints what pdSTL returns. In particular there is no independence
assumption anywhere -- conjunctions are resolved by the dependence-agnostic
Frechet rules, never by multiplying probabilities.

Each formula is evaluated on **all three backends** -- the reference
interpreter, the compiled graph, and the recurrent evaluator -- against
bit-identical atomic inputs, and agreement is asserted rather than assumed.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from models.dynamics import bound_controls, linear_gaussian_rollout, linear_system
from pdstl import (
    Always,
    And,
    Eventually,
    Predicate,
    TensorProbabilitySource,
    compile_formula,
    compile_recurrent_formula,
    evaluate,
)
from pdstl.gaussian import GaussianHalfspace, gaussian_atom_traces
from utils import load_config
from visualization.verification import (
    plot_always_verification,
    plot_eventually_verification,
    plot_stochastic_forward,
    plot_stochastic_optimization,
)

__all__ = [
    "backend_bounds",
    "build_always_scenario",
    "build_eventually_scenario",
    "build_stochastic_scenario",
    "run_always",
    "run_eventually",
    "run_stochastic_forward",
    "run_stochastic_optimization",
    "run_zero_gradient_diagnostic",
]

# float64 throughout: these are verification examples, and the three-backend
# comparison should not be limited by float32 noise.
DTYPE = torch.float64
FIGURE_DIR = "figures/verification"
BACKEND_TOL = 1e-10


def _config(section):
    return load_config("configs/stl_demos.yaml")[section]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def gaussian_trace_1d(mu, sigma):
    """Pack a scalar Gaussian belief into the provider's ``[B, T, D]`` contract."""
    mean = torch.as_tensor(np.asarray(mu), dtype=DTYPE).view(1, -1, 1)
    covariance = torch.as_tensor(np.asarray(sigma) ** 2, dtype=DTYPE).view(1, -1, 1, 1)
    return mean, covariance


def halfspace_trace(mean, covariance, halfspaces):
    """Atomic event probabilities for ``halfspaces``, as ``{uid: [B, T, 2]}``."""
    return gaussian_atom_traces(mean, covariance, halfspaces)


def backend_bounds(formula, traces, horizon, *, time=0, tol=BACKEND_TOL):
    """Evaluate ``formula`` on all three backends and require them to agree.

    The reference interpreter reads the *same* atomic tensors as the two tensor
    backends via :class:`~pdstl.base.TensorProbabilitySource`, so this compares
    three evaluation strategies over one set of inputs rather than three
    separately-computed sets of numbers.

    Returns ``{"reference": [l, u], "compiled": [...], "recurrent": [...]}``.
    """
    source = TensorProbabilitySource(traces, horizon=horizon)
    reference = evaluate(formula, source)
    compiled = compile_formula(formula, horizon=horizon)(traces)
    recurrent = compile_recurrent_formula(formula, horizon=horizon)(traces)

    if not torch.allclose(compiled, reference, atol=tol):
        raise AssertionError(f"compiled != reference for {formula}\n{compiled}\n{reference}")
    if not torch.allclose(recurrent, reference, atol=tol):
        raise AssertionError(f"recurrent != reference for {formula}\n{recurrent}\n{reference}")

    return {
        "reference": reference[0, time].tolist(),
        "compiled": compiled[0, time].tolist(),
        "recurrent": recurrent[0, time].tolist(),
    }


def print_backend_table(title, bounds_by_backend):
    """Print the ``backend / lower / upper`` table for one formula."""
    print(f"\n  {title}")
    print(f"    {'backend':<12}{'lower':>12}{'upper':>12}")
    for backend, (lower, upper) in bounds_by_backend.items():
        print(f"    {backend:<12}{lower:>12.6f}{upper:>12.6f}")


def _standard_normal_cdf(z):
    """``Phi(z)`` for a float, used only for the independent analytic check."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def rectangle_region(name, x_range, y_range):
    """A rectangle as the **conjunction of its four halfspaces**.

    Each face is a separate atomic predicate and the rectangle event is their
    logical conjunction, so pdSTL's own Frechet machinery resolves the
    intersection. No independence between faces is assumed, and no bespoke
    "rectangle probability" is computed anywhere.
    """
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    faces = [
        (f"{name}: x>={x_lo:g}", [1.0, 0.0], x_lo),
        (f"{name}: x<={x_hi:g}", [-1.0, 0.0], -x_hi),
        (f"{name}: y>={y_lo:g}", [0.0, 1.0], y_lo),
        (f"{name}: y<={y_hi:g}", [0.0, -1.0], -y_hi),
    ]

    predicates = [Predicate(name=label) for label, _, _ in faces]
    halfspaces = [
        GaussianHalfspace(
            predicate=predicate,
            normal=torch.tensor(normal, dtype=DTYPE),
            threshold=threshold,
        )
        for predicate, (_, normal, threshold) in zip(predicates, faces)
    ]
    # Balanced tree rather than a left fold: identical semantics, and it keeps
    # the printed formula readable.
    formula = And(And(predicates[0], predicates[1]), And(predicates[2], predicates[3]))
    return formula, halfspaces


# ---------------------------------------------------------------------------
# Verification A -- Eventually: temporal union
# ---------------------------------------------------------------------------


def build_eventually_scenario(config=None):
    """``F[5,10](x >= 8)`` over a scalar linear-Gaussian trace.

    The belief trace comes from the existing
    :func:`models.dynamics.linear_system` propagator, so the state model is
    shared with the introductory examples rather than reinvented here.
    """
    config = config or _config("verification_a")
    horizon = int(config["horizon"])
    times = np.arange(horizon + 1, dtype=float)

    mean_trace, var_trace = linear_system(
        a=config["a"],
        b=config["b"],
        g=config["g"],
        q=config["q"],
        mu=config["mu"],
        P=config["P"],
        t=times,
        control_func=lambda _t: config["control"],
    )
    sigma_trace = np.sqrt(var_trace)

    mean, covariance = gaussian_trace_1d(mean_trace, sigma_trace)
    predicate = Predicate(name=f"x >= {config['threshold']:g}")
    halfspace = GaussianHalfspace(
        predicate=predicate,
        normal=torch.tensor([1.0], dtype=DTYPE),
        threshold=config["threshold"],
    )
    traces = halfspace_trace(mean, covariance, [halfspace])

    a, b = (int(v) for v in config["interval"])
    formula = Eventually(predicate, interval=[a, b])

    return {
        "config": config,
        "horizon": horizon,
        "times": times,
        "mean_trace": mean_trace,
        "sigma_trace": sigma_trace,
        "predicate": predicate,
        "formula": formula,
        "interval": (a, b),
        "traces": traces,
        "atom_probabilities": traces[predicate.uid][0, :, 0].numpy(),
    }


def analytic_eventually_bounds(atom_probabilities, interval):
    """The union rule written out directly, for an independent check.

    ``lower = max_k p_k`` and ``upper = min(1, sum_k p_k)`` over the window --
    what the hard semantics must reproduce for exact (degenerate) atomic
    probabilities.
    """
    a, b = interval
    window = np.asarray(atom_probabilities)[a : b + 1]
    return float(window.max()), float(min(1.0, window.sum()))


def run_eventually(plot=True):
    """Verification A: atomic event probabilities -> temporal union."""
    scenario = build_eventually_scenario()
    a, b = scenario["interval"]
    probabilities = scenario["atom_probabilities"]

    print(f"\n  formula: {scenario['formula']}")
    print(f"  {'k':>3}{'mu_k':>10}{'sigma_k':>10}{'p_k = P(x_k >= 8)':>20}")
    for k in range(scenario["horizon"] + 1):
        marker = " <-- window" if a <= k <= b else ""
        print(
            f"  {k:>3}{scenario['mean_trace'][k]:>10.3f}"
            f"{scenario['sigma_trace'][k]:>10.3f}{probabilities[k]:>20.6f}{marker}"
        )

    bounds = backend_bounds(scenario["formula"], scenario["traces"], scenario["horizon"])
    print_backend_table(f"F[{a},{b}](x >= 8)", bounds)

    expected_lower, expected_upper = analytic_eventually_bounds(probabilities, (a, b))
    argmax = int(np.argmax(probabilities[a : b + 1])) + a
    print(f"\n    union rule:  lower = max_k p_k          = {expected_lower:.6f}  (attained at k={argmax})")
    print(f"                 upper = min(1, sum_k p_k)  = {expected_upper:.6f}")
    print(f"    sum_k p_k over the window = {probabilities[a:b+1].sum():.6f} "
          f"({'saturated at 1' if probabilities[a:b+1].sum() >= 1 else 'below 1, so the upper bound is informative'})")

    scenario["bounds"] = bounds
    scenario["argmax_time"] = argmax
    if plot:
        scenario["figure"] = plot_eventually_verification(scenario, FIGURE_DIR)
        print(f"\n    figure: {scenario['figure']}")
    return scenario


# ---------------------------------------------------------------------------
# Verification B -- Always: conjunction then temporal intersection
# ---------------------------------------------------------------------------


def build_always_scenario(config=None):
    """``G[2,6]((z > 3) AND (z < 5))`` over a scalar Gaussian trace.

    The band is represented *logically*, as the conjunction of two halfspace
    predicates on the same random variable, so the conjunction is resolved by
    pdSTL rather than by a bespoke interval predicate.
    """
    config = config or _config("verification_b")
    mu_trace = np.asarray(config["mu"], dtype=float)
    sigma_trace = np.asarray(config["sigma"], dtype=float)
    horizon = len(mu_trace) - 1
    low, high = float(config["lower_threshold"]), float(config["upper_threshold"])

    mean, covariance = gaussian_trace_1d(mu_trace, sigma_trace)
    mu_low = Predicate(name=f"z > {low:g}")
    mu_high = Predicate(name=f"z < {high:g}")
    halfspaces = [
        GaussianHalfspace(predicate=mu_low, normal=torch.tensor([1.0], dtype=DTYPE), threshold=low),
        GaussianHalfspace(predicate=mu_high, normal=torch.tensor([-1.0], dtype=DTYPE), threshold=-high),
    ]
    traces = halfspace_trace(mean, covariance, halfspaces)

    band = And(mu_low, mu_high)
    a, b = (int(v) for v in config["interval"])
    formula = Always(band, interval=[a, b])

    # The band enclosure at every time, straight from the hard semantics.
    band_trace = compile_formula(band, horizon=horizon)(traces)[0]

    return {
        "config": config,
        "horizon": horizon,
        "times": np.arange(horizon + 1),
        "mu_trace": mu_trace,
        "sigma_trace": sigma_trace,
        "thresholds": (low, high),
        "interval": (a, b),
        "band": band,
        "formula": formula,
        "traces": traces,
        "p_low": traces[mu_low.uid][0, :, 0].numpy(),
        "p_high": traces[mu_high.uid][0, :, 0].numpy(),
        "band_lower": band_trace[:, 0].numpy(),
        "band_upper": band_trace[:, 1].numpy(),
        "analytic_band": analytic_band_probability(mu_trace, sigma_trace, low, high),
    }


def analytic_band_probability(mu_trace, sigma_trace, low, high):
    """``P(low < Z < high)`` in closed form -- an EXTERNAL sanity reference.

    This is *not* fed into pdSTL and is *not* the pdSTL bound. It is available
    only because the example uses one scalar Gaussian; pdSTL deliberately works
    from the two supplied event probabilities plus its dependence-agnostic
    rules, and knows nothing about the joint law behind them.
    """
    return np.array(
        [
            _standard_normal_cdf((high - m) / s) - _standard_normal_cdf((low - m) / s)
            for m, s in zip(np.asarray(mu_trace), np.asarray(sigma_trace))
        ]
    )


def run_always(plot=True):
    """Verification B: conjunction -> temporal intersection -> Frechet bounds."""
    scenario = build_always_scenario()
    a, b = scenario["interval"]
    low, high = scenario["thresholds"]
    product = scenario["p_low"] * scenario["p_high"]

    print(f"\n  formula: {scenario['formula']}")
    print(
        f"  {'k':>3}{'mu_k':>8}{'sig_k':>7}{'P(z>3)':>10}{'P(z<5)':>10}"
        f"{'band L':>10}{'band U':>10}{'exact':>10}{'product':>10}"
    )
    for k in range(scenario["horizon"] + 1):
        marker = " <--" if a <= k <= b else ""
        print(
            f"  {k:>3}{scenario['mu_trace'][k]:>8.2f}{scenario['sigma_trace'][k]:>7.2f}"
            f"{scenario['p_low'][k]:>10.6f}{scenario['p_high'][k]:>10.6f}"
            f"{scenario['band_lower'][k]:>10.6f}{scenario['band_upper'][k]:>10.6f}"
            f"{scenario['analytic_band'][k]:>10.6f}{product[k]:>10.6f}{marker}"
        )

    print("\n    'exact' is the analytic P(3<z<5) for this scalar Gaussian: an EXTERNAL")
    print("    reference, never an input to pdSTL. 'product' is what an (incorrect)")
    print("    independence assumption would give -- shown only to be rejected.")
    band_error = np.abs(scenario["band_lower"] - scenario["analytic_band"]).max()
    product_error = np.abs(product - scenario["analytic_band"]).max()
    print(f"\n    max |Frechet band lower - exact|  = {band_error:.3e}   <- the conjunction is exact")
    print(f"    max |independence product - exact| = {product_error:.3e}   <- and is NOT a product")

    bounds = backend_bounds(scenario["formula"], scenario["traces"], scenario["horizon"])
    print_backend_table(f"G[{a},{b}]((z > {low:g}) AND (z < {high:g}))", bounds)

    window_lower = scenario["band_lower"][a : b + 1]
    window_upper = scenario["band_upper"][a : b + 1]
    m = len(window_lower)  # post-reduction operand count: the window events are all distinct
    print(f"\n    temporal intersection over k = {a}..{b}, m = {m} distinct events")
    print(f"                 lower = max(0, sum L_k - (m-1)) = max(0, {window_lower.sum():.6f} - {m - 1}) "
          f"= {max(0.0, window_lower.sum() - (m - 1)):.6f}")
    print(f"                 upper = min_k U_k               = {window_upper.min():.6f}")

    scenario["bounds"] = bounds
    scenario["product"] = product
    if plot:
        scenario["figure"] = plot_always_verification(scenario, FIGURE_DIR)
        print(f"\n    figure: {scenario['figure']}")
    return scenario


# ---------------------------------------------------------------------------
# Verification C -- complete stochastic-system pipeline
# ---------------------------------------------------------------------------


def build_stochastic_scenario(config=None, goal_key="goal_region"):
    """``G[0,N] Safe AND F[a,N] Goal`` on a 2-D linear stochastic system.

    ``goal_key`` selects which configured rectangle is the goal; the C3
    diagnostic reuses this same builder with the unreachable one.
    """
    config = config or _config("verification_c")
    horizon = int(config["horizon"])
    dim = len(config["x0_mean"])

    safe, safe_halfspaces = rectangle_region(
        "safe", config["safe_region"]["x"], config["safe_region"]["y"]
    )
    goal, goal_halfspaces = rectangle_region(
        "goal", config[goal_key]["x"], config[goal_key]["y"]
    )

    goal_a, goal_b = (int(v) for v in config["goal_interval"])
    phi_safe = Always(safe, interval=[0, horizon])
    phi_goal = Eventually(goal, interval=[goal_a, goal_b])
    phi = And(phi_safe, phi_goal)

    return {
        "config": config,
        "horizon": horizon,
        "dim": dim,
        "A": torch.eye(dim, dtype=DTYPE),
        "B": float(config["dt"]) * torch.eye(dim, dtype=DTYPE),
        "Q": torch.eye(dim, dtype=DTYPE) * float(config["q_std"]) ** 2,
        "x0_mean": torch.tensor(config["x0_mean"], dtype=DTYPE),
        "x0_cov": torch.zeros(dim, dim, dtype=DTYPE),
        "safe": safe,
        "goal": goal,
        "phi_safe": phi_safe,
        "phi_goal": phi_goal,
        "phi": phi,
        "halfspaces": safe_halfspaces + goal_halfspaces,
        "safe_halfspaces": safe_halfspaces,
        "goal_halfspaces": goal_halfspaces,
        "goal_interval": (goal_a, goal_b),
        "safe_region": config["safe_region"],
        "goal_region": config[goal_key],
    }


def initial_controls(scenario, requires_grad=False):
    """The hand-designed open-loop control parameters for C1/C2.

    Returned in unconstrained (pre-``tanh``) coordinates, since that is what the
    optimizer in C2 actually varies.
    """
    config = scenario["config"]
    u_max = float(config["u_max"])
    v = torch.zeros(scenario["horizon"], scenario["dim"], dtype=DTYPE)
    v[:, 0] = math.atanh(float(config["u_init"]) / u_max)
    return v.requires_grad_(requires_grad)


def propagate(scenario, v):
    """Controls -> belief -> atomic event probabilities."""
    controls = bound_controls(v, float(scenario["config"]["u_max"]))
    mean, covariance = linear_gaussian_rollout(
        controls,
        scenario["x0_mean"],
        scenario["x0_cov"],
        scenario["A"],
        scenario["B"],
        scenario["Q"],
    )
    traces = halfspace_trace(mean, covariance, scenario["halfspaces"])
    return mean, covariance, traces


def region_bounds_trace(scenario, traces, which):
    """Per-time enclosure of the Safe / Goal *region* event."""
    formula = scenario[which]
    return compile_formula(formula, horizon=scenario["horizon"])(traces)[0]


def run_stochastic_forward(plot=True):
    """Verification C1: forward pipeline on a fixed hand-designed control."""
    scenario = build_stochastic_scenario()
    v = initial_controls(scenario)
    mean, covariance, traces = propagate(scenario, v)

    horizon = scenario["horizon"]
    safe_trace = region_bounds_trace(scenario, traces, "safe")
    goal_trace = region_bounds_trace(scenario, traces, "goal")

    print(f"\n  system:  x_(k+1) = A x_k + B u_k + w_k,  w_k ~ N(0, Q),  N = {horizon}")
    print(f"  safe region: x in {scenario['safe_region']['x']}, y in {scenario['safe_region']['y']}")
    print(f"  goal region: x in {scenario['goal_region']['x']}, y in {scenario['goal_region']['y']}")
    print(f"  final mean state: ({mean[0, -1, 0]:.3f}, {mean[0, -1, 1]:.3f}), "
          f"sigma = {covariance[0, -1, 0, 0].sqrt():.3f}")

    results = {}
    for title, key in (
        ("G Safe", "phi_safe"),
        ("F Goal", "phi_goal"),
        ("G Safe AND F Goal", "phi"),
    ):
        results[title] = backend_bounds(scenario[key], traces, horizon)

    print(f"\n  {'formula':<22}{'lower':>12}{'upper':>12}")
    for title, bounds in results.items():
        lower, upper = bounds["compiled"]
        print(f"  {title:<22}{lower:>12.6f}{upper:>12.6f}")
    print("\n  backend agreement (all formulas):  reference == compiled == recurrent  ✓"
          f"  (atol {BACKEND_TOL:g})")

    goal_a, goal_b = scenario["goal_interval"]
    print(f"\n    atomic safety probability  min = {safe_trace[:, 0].min():.6f}  "
          f"max = {safe_trace[:, 0].max():.6f}   (lower bounds, k = 0..{horizon})")
    print(f"    atomic goal probability    min = {goal_trace[goal_a:goal_b + 1, 0].min():.6f}  "
          f"max = {goal_trace[goal_a:goal_b + 1, 0].max():.6f}   (lower bounds, k = {goal_a}..{goal_b})")

    lower_g = results["G Safe"]["compiled"][0]
    lower_f = results["F Goal"]["compiled"][0]
    print(f"\n    top-level conjunction pre-clamp: L_G + L_F - 1 = {lower_g + lower_f - 1:+.6f}  "
          f"({'ACTIVE branch' if lower_g + lower_f - 1 > 0 else 'clamped to zero'})")

    scenario.update(
        v=v, mean=mean, covariance=covariance, traces=traces,
        safe_trace=safe_trace, goal_trace=goal_trace, results=results,
    )
    if plot:
        scenario["figure"] = plot_stochastic_forward(scenario, FIGURE_DIR)
        print(f"\n    figure: {scenario['figure']}")
    return scenario


def optimize_hard_lower_bound(scenario, iterations=None, learning_rate=None):
    """Maximize the exact hard pdSTL lower probability bound over the controls.

    The objective is ``-P_lower(phi)`` where ``P_lower`` is the hard Frechet
    lower bound itself -- not a smoothed surrogate, not a robustness margin.
    Gradients reach the controls through the recurrent evaluator, the Gaussian
    provider, and the belief propagation.

    The controls are box-constrained via :func:`models.dynamics.bound_controls`;
    that squashing is an optimizer/application choice and forms no part of the
    pdSTL semantics.
    """
    settings = scenario["config"]["optimizer"]
    iterations = int(iterations if iterations is not None else settings["iterations"])
    learning_rate = float(
        learning_rate if learning_rate is not None else settings["learning_rate"]
    )

    v = initial_controls(scenario, requires_grad=True)
    evaluator = compile_recurrent_formula(scenario["phi"], horizon=scenario["horizon"])
    optimizer = torch.optim.Adam([v], lr=learning_rate)

    history = {"lower": [], "upper": [], "loss": [], "grad_norm": []}
    initial_state = None

    for iteration in range(iterations):
        optimizer.zero_grad()
        mean, covariance, traces = propagate(scenario, v)
        out = evaluator(traces)
        lower, upper = out[0, 0, 0], out[0, 0, 1]

        loss = -lower
        loss.backward()
        grad_norm = v.grad.norm().item()

        history["lower"].append(lower.item())
        history["upper"].append(upper.item())
        history["loss"].append(loss.item())
        history["grad_norm"].append(grad_norm)

        if iteration == 0:
            initial_state = (mean.detach().clone(), covariance.detach().clone())

        optimizer.step()

    with torch.no_grad():
        final_mean, final_covariance, _ = propagate(scenario, v)

    return {
        "v": v.detach(),
        "history": history,
        "initial_mean": initial_state[0],
        "initial_covariance": initial_state[1],
        "final_mean": final_mean,
        "final_covariance": final_covariance,
        "iterations": iterations,
        "learning_rate": learning_rate,
    }


def run_stochastic_optimization(plot=True, iterations=None):
    """Verification C2: differentiability and direct hard-bound optimization."""
    scenario = build_stochastic_scenario()
    result = optimize_hard_lower_bound(scenario, iterations=iterations)
    history = result["history"]

    print("\n  objective: maximize the hard pdSTL lower probability bound P_lower(phi)")
    print("  formula:   G Safe AND F Goal")
    print(f"  optimizer: Adam, lr = {result['learning_rate']}, {result['iterations']} iterations")
    print(f"\n  {'iter':>6}{'P_lower':>12}{'P_upper':>12}{'loss':>12}{'grad norm':>14}")
    step = max(1, result["iterations"] // 10)
    for i in list(range(0, result["iterations"], step)) + [result["iterations"] - 1]:
        print(
            f"  {i:>6}{history['lower'][i]:>12.6f}{history['upper'][i]:>12.6f}"
            f"{history['loss'][i]:>12.6f}{history['grad_norm'][i]:>14.6e}"
        )

    improvement = history["lower"][-1] - history["lower"][0]
    print(f"\n    P_lower:  {history['lower'][0]:.6f} -> {history['lower'][-1]:.6f}  ({improvement:+.6f})")
    print(f"    P_upper:  {history['upper'][0]:.6f} -> {history['upper'][-1]:.6f}")
    print(f"    gradient: {history['grad_norm'][0]:.6e} -> {history['grad_norm'][-1]:.6e}")
    print(f"    improved: {improvement > 0}")

    scenario["optimization"] = result
    if plot:
        scenario["figure"] = plot_stochastic_optimization(scenario, result, FIGURE_DIR)
        print(f"\n    figure: {scenario['figure']}")
    return scenario, result


def run_zero_gradient_diagnostic():
    """Verification C3: the expected hard zero-gradient plateau, reported not fixed.

    Same system and formula as C1/C2, with the goal moved out of reach. The
    Frechet pre-clamp expression for the top-level conjunction goes negative,
    so ``P_lower`` is exactly 0 and its gradient is exactly 0.

    This is the exact hard semantics behaving as specified. It is what
    distinguishes "the end-to-end differentiable pipeline works" -- which this
    suite demonstrates -- from "the hard lower semantics is globally
    gradient-rich", which it is not and which is not claimed. No smoothing,
    margin, or surrogate is introduced to hide it.
    """
    scenario = build_stochastic_scenario(goal_key="unreachable_goal")
    v = initial_controls(scenario, requires_grad=True)
    _, _, traces = propagate(scenario, v)
    horizon = scenario["horizon"]

    lower_g = compile_recurrent_formula(scenario["phi_safe"], horizon=horizon)(traces)[0, 0, 0]
    lower_f = compile_recurrent_formula(scenario["phi_goal"], horizon=horizon)(traces)[0, 0, 0]
    out = compile_recurrent_formula(scenario["phi"], horizon=horizon)(traces)
    out[0, 0, 0].backward()

    pre_clamp = (lower_g + lower_f - 1).item()
    grad_norm = v.grad.norm().item()

    print(f"\n  goal region moved out of reach: x in {scenario['goal_region']['x']}")
    print(f"\n    L_G (G Safe)                = {lower_g.item():.6f}")
    print(f"    L_F (F Goal)                = {lower_f.item():.6f}")
    print(f"    Frechet pre-clamp L_G+L_F-1 = {pre_clamp:+.6f}   ({'< 0' if pre_clamp < 0 else '>= 0'})")
    print(f"    P_lower                     = {out[0, 0, 0].item():.6f}")
    print(f"    P_upper                     = {out[0, 0, 1].item():.6f}")
    print(f"    ||grad P_lower||            = {grad_norm:.6e}")
    print("\n    EXPECTED behavior of the exact hard semantics: below the clamp the lower")
    print("    bound is identically zero, so it carries no gradient. Reported, not fixed --")
    print("    no margin, no smoothing, no surrogate.")

    return {
        "lower_g": lower_g.item(),
        "lower_f": lower_f.item(),
        "pre_clamp": pre_clamp,
        "p_lower": out[0, 0, 0].item(),
        "p_upper": out[0, 0, 1].item(),
        "grad_norm": grad_norm,
    }
