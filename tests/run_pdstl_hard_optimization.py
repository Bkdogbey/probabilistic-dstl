"""Diagnostic: raw (unsmoothed) Adam optimization against the exact hard
pdSTL lower bound, on four formulas of increasing difficulty.

Not a pytest file (no test_ prefix, not collected). Run directly:

    PYTHONPATH=src python tests/run_pdstl_hard_optimization.py

This is diagnostic, not prescriptive: it uses compile_formula/CompiledFormula
(the exact compiled hard-semantics graph, no smoothing, no scale parameter)
and J = -lower_bound only, on purpose, to observe how the raw Frechet
lower/upper bounds behave under gradient-based optimization -- including any
zero-gradient dead zones -- rather than to demonstrate a working planner. See
the plan's Phase 6-8 for the full experimental design. Findings are reported,
not remedied, here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from pdstl import Always, And, Eventually, Predicate, compile_formula
from pdstl.gaussian import GaussianHalfspace, gaussian_atom_traces

from pdstl_rollout import differentiable_rollout

# ---------------------------------------------------------------------------
# Regime-classification thresholds
# ---------------------------------------------------------------------------

ZERO_GRAD_TOL = 1e-8
BOUND_IMPROVEMENT_TOL = 1e-4
SPARSE_FRAC_THRESHOLD = 0.3
TAIL_SAT_THRESHOLD = 1e-4

DT = 1.0
U_MAX = 1.0
Q_STD = 0.3


@dataclass
class IterationLog:
    iteration: int
    objective: float
    lower_bound: float
    upper_bound: float
    grad_norm: float
    grad_nonzero_frac: float
    atom_p_min: float
    atom_p_max: float
    atom_p_mean: float
    tail_proxy: float


@dataclass
class CaseResult:
    name: str
    log: list[IterationLog] = field(default_factory=list)
    final_v: torch.Tensor | None = None
    final_mean: torch.Tensor | None = None

    def classify(self) -> str:
        if not self.log:
            return "other"

        finite = all(
            math.isfinite(entry.grad_norm)
            and math.isfinite(entry.lower_bound)
            and math.isfinite(entry.upper_bound)
            for entry in self.log
        )
        monotone_bounds = all(entry.lower_bound <= entry.upper_bound + 1e-6 for entry in self.log)
        if not finite or not monotone_bounds:
            return "numerical issue"

        n = len(self.log)
        trailing = self.log[-max(1, n // 10) :]
        trailing_grad_norm = sum(entry.grad_norm for entry in trailing) / len(trailing)

        if (
            self.log[0].grad_norm < ZERO_GRAD_TOL
            and trailing_grad_norm < ZERO_GRAD_TOL
            and self.log[0].lower_bound == 0.0
        ):
            return "zero-gradient saturation"

        # Checked before the saturation/sparsity diagnostics below: a run
        # that substantially improved the bound while the gradient was
        # nonzero for most of its course is a success, even though -- like
        # any bounded objective approaching its supremum -- it will show a
        # naturally shrinking gradient and tail-saturated atoms near the end.
        # That end-state is expected arithmetic, not a pathology to flag.
        nonzero_grad_iters = sum(1 for entry in self.log if entry.grad_norm >= ZERO_GRAD_TOL)
        improved = self.log[-1].lower_bound - self.log[0].lower_bound > BOUND_IMPROVEMENT_TOL
        mostly_nonzero = nonzero_grad_iters > 0.7 * n
        if improved and mostly_nonzero:
            return "exact-hard works"

        sparse_iters = sum(1 for entry in self.log if entry.grad_nonzero_frac < SPARSE_FRAC_THRESHOLD)
        if sparse_iters > n / 2 and nonzero_grad_iters > n / 2:
            return "sparse max gradient"

        tail_iters = sum(1 for entry in self.log if entry.tail_proxy < TAIL_SAT_THRESHOLD)
        if tail_iters > n / 2:
            return "upstream Gaussian saturation"

        return "other"


@dataclass
class Scenario:
    name: str
    N: int
    D: int
    build_formula_and_halfspaces: callable  # () -> (formula, list[GaussianHalfspace])
    v_init: callable  # () -> Tensor[N, D]


def _process_noise(D: int) -> torch.Tensor:
    return torch.eye(D) * Q_STD**2


def _rollout(v: torch.Tensor, D: int) -> tuple[torch.Tensor, torch.Tensor]:
    x0_mean = torch.zeros(D)
    x0_cov = torch.zeros(D, D)
    return differentiable_rollout(v, x0_mean, x0_cov, dt=DT, u_max=U_MAX, process_noise=_process_noise(D))


def _stochastic_tail_proxy(
    traces: dict[int, torch.Tensor],
    covariance: torch.Tensor,
    halfspaces: list[GaussianHalfspace],
) -> float:
    """``min(p, 1-p)`` over atom-time entries whose underlying variance is not
    (numerically) zero.

    Excludes deterministic-branch entries (e.g. a fixed, zero-variance start
    state at k=0): those trivially register as "saturated" regardless of the
    trajectory and are a different, expected phenomenon (a fixed initial
    condition), not evidence of the Gaussian CDF's tail flattening as the
    trajectory moves deep into a region -- which is what this proxy is meant
    to detect.
    """
    values = []
    for halfspace in halfspaces:
        normal = halfspace.normal.to(dtype=covariance.dtype)
        v_r = torch.einsum("d,btde,e->bt", normal, covariance, normal)
        p = traces[halfspace.predicate.uid][..., 0]
        stochastic = v_r > 1e-9
        if bool(stochastic.any()):
            selected = p[stochastic]
            values.append(torch.minimum(selected, 1.0 - selected))
    if not values:
        return 1.0  # no stochastic entries at all; nothing to saturate
    return torch.cat(values).min().item()


def run_case(scenario: Scenario, *, n_iters: int, lr: float, reg_weight: float = 0.0) -> CaseResult:
    v = scenario.v_init().clone().requires_grad_(True)
    formula, halfspaces = scenario.build_formula_and_halfspaces()
    compiled = compile_formula(formula, horizon=scenario.N)

    optimizer = torch.optim.Adam([v], lr=lr)
    result = CaseResult(name=scenario.name)

    for iteration in range(n_iters):
        optimizer.zero_grad()

        mean, covariance = _rollout(v, scenario.D)
        traces = gaussian_atom_traces(mean, covariance, halfspaces)
        out = compiled(traces)
        lower_bound = out[0, 0, 0]
        upper_bound = out[0, 0, 1]

        objective = -lower_bound
        if reg_weight > 0.0:
            objective = objective + reg_weight * (v**2).sum()

        objective.backward()

        grad = v.grad.detach()
        grad_norm = grad.norm().item()
        grad_nonzero_frac = (grad.abs() > 1e-12).float().mean().item()

        all_p = torch.cat([trace[..., 0].reshape(-1) for trace in traces.values()])
        atom_p_min = all_p.min().item()
        atom_p_max = all_p.max().item()
        atom_p_mean = all_p.mean().item()
        tail_proxy = _stochastic_tail_proxy(traces, covariance, halfspaces)

        result.log.append(
            IterationLog(
                iteration=iteration,
                objective=objective.item(),
                lower_bound=lower_bound.item(),
                upper_bound=upper_bound.item(),
                grad_norm=grad_norm,
                grad_nonzero_frac=grad_nonzero_frac,
                atom_p_min=atom_p_min,
                atom_p_max=atom_p_max,
                atom_p_mean=atom_p_mean,
                tail_proxy=tail_proxy,
            )
        )

        optimizer.step()

    result.final_v = v.detach().clone()
    result.final_mean, _ = _rollout(v.detach(), scenario.D)
    return result


# ---------------------------------------------------------------------------
# Cases A-D
# ---------------------------------------------------------------------------


def case_a() -> Scenario:
    def build():
        mu_goal = Predicate(name="mu_goal")
        hs = GaussianHalfspace(predicate=mu_goal, normal=torch.tensor([1.0, 0.0]), threshold=0.6)
        return Always(mu_goal, interval=[5, 5]), [hs]

    return Scenario("A: atomic (goal, singleton)", N=5, D=2, build_formula_and_halfspaces=build,
                     v_init=lambda: torch.zeros(5, 2))


def case_b() -> Scenario:
    def build():
        mu_goal = Predicate(name="mu_goal")
        hs = GaussianHalfspace(predicate=mu_goal, normal=torch.tensor([1.0, 0.0]), threshold=0.6)
        return Eventually(mu_goal, interval=[0, 5]), [hs]

    return Scenario("B: Eventually (goal)", N=5, D=2, build_formula_and_halfspaces=build,
                     v_init=lambda: torch.zeros(5, 2))


def case_c1() -> Scenario:
    def build():
        mu_safe = Predicate(name="mu_safe")
        hs = GaussianHalfspace(predicate=mu_safe, normal=torch.tensor([0.0, 1.0]), threshold=-0.6)
        return Always(mu_safe, interval=[0, 2]), [hs]

    return Scenario("C1: Always (safety, positive lower)", N=2, D=2, build_formula_and_halfspaces=build,
                     v_init=lambda: torch.zeros(2, 2))


def case_c2() -> Scenario:
    def build():
        mu_safe = Predicate(name="mu_safe")
        hs = GaussianHalfspace(predicate=mu_safe, normal=torch.tensor([0.0, 1.0]), threshold=-0.6)
        return Always(mu_safe, interval=[0, 9]), [hs]

    return Scenario("C2: Always (safety, Frechet zero region)", N=9, D=2,
                     build_formula_and_halfspaces=build, v_init=lambda: torch.zeros(9, 2))


def case_d() -> Scenario:
    def build():
        mu_safe = Predicate(name="mu_safe")
        mu_goal = Predicate(name="mu_goal")
        hs_safe = GaussianHalfspace(predicate=mu_safe, normal=torch.tensor([0.0, 1.0]), threshold=-0.8)
        hs_goal = GaussianHalfspace(predicate=mu_goal, normal=torch.tensor([1.0, 0.0]), threshold=0.3)
        formula = And(Always(mu_safe, interval=[0, 3]), Eventually(mu_goal, interval=[0, 3]))
        return formula, [hs_safe, hs_goal]

    return Scenario("D: mixed (safety & eventually goal)", N=3, D=2, build_formula_and_halfspaces=build,
                     v_init=lambda: torch.zeros(3, 2))


def case_d_zero_region() -> Scenario:
    """Optional: same formula as D, smaller margins land it in the zero region."""

    def build():
        mu_safe = Predicate(name="mu_safe")
        mu_goal = Predicate(name="mu_goal")
        hs_safe = GaussianHalfspace(predicate=mu_safe, normal=torch.tensor([0.0, 1.0]), threshold=-0.6)
        hs_goal = GaussianHalfspace(predicate=mu_goal, normal=torch.tensor([1.0, 0.0]), threshold=0.6)
        formula = And(Always(mu_safe, interval=[0, 3]), Eventually(mu_goal, interval=[0, 3]))
        return formula, [hs_safe, hs_goal]

    return Scenario("D': mixed (Frechet zero region via top-level And)", N=3, D=2,
                     build_formula_and_halfspaces=build, v_init=lambda: torch.zeros(3, 2))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(result: CaseResult) -> None:
    first = result.log[0]
    last = result.log[-1]
    grad_went_zero = last.grad_norm < ZERO_GRAD_TOL
    improved = last.lower_bound - first.lower_bound > BOUND_IMPROVEMENT_TOL

    print(f"\n=== {result.name} ===")
    print(f"  iterations:           {len(result.log)}")
    print(f"  initial lower bound:  {first.lower_bound:.6f}")
    print(f"  final lower bound:    {last.lower_bound:.6f}")
    print(f"  initial upper bound:  {first.upper_bound:.6f}")
    print(f"  final upper bound:    {last.upper_bound:.6f}")
    print(f"  initial grad norm:    {first.grad_norm:.6e}")
    print(f"  final grad norm:      {last.grad_norm:.6e}")
    print(f"  bound improved:       {improved}")
    print(f"  gradient went zero:   {grad_went_zero}")
    print(f"  final v (flattened):  {result.final_v.flatten().tolist()}")
    print(f"  final mean trajectory: {result.final_mean[0].tolist()}")
    print(f"  CLASSIFICATION:       {result.classify()}")


def main() -> None:
    torch.manual_seed(20260825)

    scenarios = [case_a(), case_b(), case_c1(), case_c2(), case_d(), case_d_zero_region()]
    results = []
    for scenario in scenarios:
        n_iters = 300
        result = run_case(scenario, n_iters=n_iters, lr=0.05)
        print_report(result)
        results.append(result)

    print("\n=== Summary ===")
    for result in results:
        first, last = result.log[0], result.log[-1]
        print(
            f"{result.name:45s} lower {first.lower_bound:.4f} -> {last.lower_bound:.4f}  "
            f"grad {first.grad_norm:.2e} -> {last.grad_norm:.2e}  [{result.classify()}]"
        )


if __name__ == "__main__":
    main()
