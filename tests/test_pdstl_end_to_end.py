"""The first end-to-end differentiable pdSTL pipeline.

    v -> bounded controls -> Gaussian mean/covariance rollout
      -> gaussian_atom_traces -> compile_formula -> CompiledFormula
      -> lower satisfaction bound

This test uses compile_formula/CompiledFormula exclusively, never
propagate.evaluate: the compiled graph is the synthesis execution path under
test here, not the semantic oracle (that role belongs to propagate.py, and
is not what this file is checking).
"""

import torch

from pdstl import Always, Predicate, compile_formula
from pdstl.gaussian import GaussianHalfspace, gaussian_atom_traces

from tests.pdstl_rollout import differentiable_rollout

# Shared scenario: N=2, y0=0, safety margin M=0.6, q_std=0.3, v=0 init.
# Hand-derived (and numerically confirmed): per-step safety probability
# [1.0 (deterministic, k=0), 0.9772, 0.9214]; Frechet-intersection lower
# ~0.8986 (margin ~0.90 from the 0 boundary) and upper ~0.9214 at k=2, with
# a ~0.056 gap over the k=1 runner-up (0.9772) -- both margins are far larger
# than the h=1e-4 finite-difference step used below, so a perturbation of
# that size cannot flip which branch of clamp/amin is active.
N = 2
D = 2
DT = 1.0
U_MAX = 1.0
Q_STD = 0.3
Y_MIN = -0.6
KINK_MARGIN_FLOOR = 0.01  # a comfortable multiple of h=1e-4


def _scenario(v):
    x0_mean = torch.zeros(D)
    x0_cov = torch.zeros(D, D)
    process_noise = torch.diag(torch.tensor([Q_STD**2, Q_STD**2]))
    mean, covariance = differentiable_rollout(
        v, x0_mean, x0_cov, dt=DT, u_max=U_MAX, process_noise=process_noise
    )

    mu_safe = Predicate(name="mu_safe")
    mu_goal = Predicate(name="mu_goal")  # unused by the formula below; for the zero-grad check
    halfspaces = [
        GaussianHalfspace(predicate=mu_safe, normal=torch.tensor([0.0, 1.0]), threshold=Y_MIN),
        GaussianHalfspace(predicate=mu_goal, normal=torch.tensor([1.0, 0.0]), threshold=999.0),
    ]
    traces = gaussian_atom_traces(mean, covariance, halfspaces)

    formula = Always(mu_safe, interval=[0, N])
    compiled = compile_formula(formula, horizon=N)
    return compiled, traces


def test_full_chain_produces_valid_bounds():
    v = torch.zeros(N, D, requires_grad=True)
    compiled, traces = _scenario(v)
    out = compiled(traces)

    assert out.shape == (1, 1, 2)
    lower, upper = out[0, 0].tolist()
    assert 0.0 <= lower <= upper <= 1.0


def test_gradient_flows_from_lower_bound_to_v():
    v = torch.zeros(N, D, requires_grad=True)
    compiled, traces = _scenario(v)
    out = compiled(traces)
    lower_bound = out[0, 0, 0]

    objective = -lower_bound
    objective.backward()

    assert v.grad is not None
    assert torch.isfinite(v.grad).all()


def test_kink_margins_are_verified_before_finite_difference_check():
    """Check, don't assume, that the chosen point avoids Frechet kinks.

    Re-derives the pre-clamp intersection argument and the winning-vs-
    runner-up amin gap directly from the same atom traces the forward pass
    uses, and asserts both margins comfortably exceed the FD step size.
    """
    v = torch.zeros(N, D, requires_grad=True)
    _, traces = _scenario(v)

    ps = [trace[0, :, 0] for trace in traces.values()]
    # The safety trace is the one that is NOT saturated at 0 (mu_goal's
    # threshold of 999 drives it to exactly 0 everywhere).
    p = next(candidate for candidate in ps if candidate.max().item() > 0.0)

    n = p.shape[0]
    pre_clamp_lower = p.sum().item() - (n - 1)
    lower_margin = pre_clamp_lower  # distance from the 0.0 clamp boundary

    sorted_p, _ = torch.sort(p)
    upper_margin = (sorted_p[1] - sorted_p[0]).item()  # winner vs. runner-up gap

    assert lower_margin > KINK_MARGIN_FLOOR, f"lower margin {lower_margin} too close to 0"
    assert upper_margin > KINK_MARGIN_FLOOR, f"upper margin {upper_margin} too close to the runner-up"


def test_finite_difference_matches_autograd_at_interior_point():
    v = torch.zeros(N, D, requires_grad=True)
    compiled, traces = _scenario(v)
    out = compiled(traces)
    lower_bound = out[0, 0, 0]
    lower_bound.backward()
    analytic = v.grad.clone()

    h = 1e-4

    def lower_at(v_value):
        compiled_h, traces_h = _scenario(v_value)
        return compiled_h(traces_h)[0, 0, 0].item()

    numeric = torch.zeros_like(v)
    base = v.detach().clone()
    for k in range(N):
        for d in range(D):
            plus = base.clone()
            plus[k, d] += h
            minus = base.clone()
            minus[k, d] -= h
            numeric[k, d] = (lower_at(plus) - lower_at(minus)) / (2 * h)

    assert torch.allclose(analytic, numeric, atol=1e-3, rtol=1e-3), (
        f"analytic={analytic}\nnumeric={numeric}"
    )

    # mu_goal is unreferenced by Always(mu_safe, ...): the x-branch (d=0)
    # gradient must be exactly zero, a free exact-answer sanity check.
    assert torch.all(analytic[:, 0] == 0.0)
    assert torch.all(numeric[:, 0] == 0.0)
