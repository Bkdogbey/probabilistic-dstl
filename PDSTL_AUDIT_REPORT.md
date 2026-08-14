# pdSTL Semantics Audit and Validation Report

This is the single canonical report for the pdSTL semantics revision. Each
validated patch has a self-contained section so it can be copied separately.
Future patches should update this file rather than create another root-level
patch report.

## Current status

| Item | Current value |
|---|---|
| Branch | `revision/stori-semantics` |
| Base commit | `bf328f579df5a0b8d762a3b98047b711f83f1c45` |
| Validated corrections | Gaussian atoms; shared Gaussian residual probability; generic rectangular Fréchet intervals; exact bounded StoRI Until |
| Current audit | 115 passed, 15 strict xfailed, 0 ordinary failed, 0 skipped |
| Existing regressions | 53 passed, 0 failed |
| Supported paper scope used here | Bounded finite-time intervals `[a,b]`, `0 <= a <= b < infinity` |

Patches 1–4 are uncommitted working-tree changes carried from
`experiments/RA-L` onto `revision/stori-semantics`. No commit or push has been
performed. The semantic-visualization task changes no production semantics.

## Audit progress

Counts below are remaining known strict expected failures, not coverage or a
percentage-correctness measure. The audit node count grew as tests were added.

| Stage | Audit pass | Strict xfail | Audit nodes |
|---|---:|---:|---:|
| Initial audit | 69 | 30 | 99 |
| After Patch 1 | 77 | 23 | 100 |
| After Patch 2 | 93 | 22 | 115 |
| After Patch 3 | 103 | 20 | 123 |
| After Patch 4/current | 115 | 15 | 130 |

## Current A–J audit matrix

| Area | Current result | Status |
|---|---|---|
| A1 documented `Belief` contract | Scalar atoms now use the contract; four planning predicates still require undeclared `mean_full`/`var_full` fields | 4 strict xfails remain |
| A2 `TorchGaussianBelief` interoperability | Its `probability_of` remains intentionally unimplemented, so core scalar atoms cannot consume it | 2 strict xfails remain |
| A3 trajectories | Offline/online indexing, suffixes, and scalar evaluation agree | Pass |
| B1 exact Gaussian atoms | Greater/Less return direct exact event probability `[p,p]` | Resolved by Patch 1 |
| B2 confidence shifts | `confidence_level` no longer shifts atomic probability | Resolved by Patch 1 |
| B3 zero variance | Deterministic equality is satisfied; negative variance is rejected | Resolved by Patches 1–2 |
| B4 tiny variance | No artificial variance floor; values and positive-variance gradients match the Gaussian reference | Resolved by Patch 2 |
| C1 independent rectangles | Generic result is a non-degenerate Fréchet interval containing the exact independent product | Resolved by Patch 3 |
| C2 correlated rectangles | Correlated SciPy joint probability lies in the dependence-agnostic generic interval | Resolved by Patch 3 |
| C3 Fréchet reference/invariants | Bounds and 100 seeded interval invariants pass | Pass |
| C4 Crazyflie rectangles | Diagonal-Gaussian product implementation is unchanged and lies in the generic interval | Pass |
| D Negation/And/Or | Exact StoRI Boolean equations and seeded invariants pass | Pass |
| E bounded Always/Eventually | Direct bounded-window reference passes | Pass |
| E4 explicit `[0,infinity]` | Always/Eventually still fail when converting infinity to `int` | 2 xfails; outside current bounded paper scope |
| F exact Until | Inclusive prefix and Fréchet candidate match the direct reference | Resolved by Patch 4 |
| G1 smoothing convergence | Always/Eventually converge to their exact reductions; smooth Until does not converge to corrected exact Until | 1 strict xfail remains |
| G2 smoothing range | Unnormalized smooth min/max can leave `[0,1]` | 3 strict xfails remain |
| G3–G4 Boolean smoothing/gradients | Positive scale leaves hard And/Or values and gradients unchanged; conjunction has a zero-gradient lower-bound region | Characterized; unresolved design issue |
| G5 planner clamp gradients | Clamp removes the STL-loss gradient below `1e-6` and above `1` | Characterized; unresolved design issue |
| H planner/config/terminology | Smooth objective vs exact selection is characterized; StoRI endpoints are still sometimes labeled as satisfaction probabilities | Terminology remains unresolved |
| I1 Monte Carlo initial covariance | Rollouts clone `x0_mean` rather than sample `x0_cov` | 1 strict xfail remains |
| I2 empirical formula equivalence | Manual metrics do not evaluate the planner's full timed STL specification | 1 strict xfail remains |
| I3 empirical confidence interval | Empirical satisfaction returns no confidence interval | 1 strict xfail remains |
| J complexity | Dense shifts and repeated Until prefix reductions exceed unconditional O(T) in general | Static finding remains |

Strict `xfail` preserves the mathematical expectation and treats an unexpected
future pass as a test failure.

---

## Patch 1 — Exact scalar Gaussian atomic predicates

### Production changes

- `src/pdstl/operators.py`: `GreaterThan` evaluates `belief.value() - threshold`;
  `LessThan` evaluates `threshold - belief.value()`. Both pass that residual
  directly to `belief.probability_of` and return `[p,p]`.
- `src/models/dynamics.py`: `GaussianBelief.probability_of` handles positive,
  zero, and negative variance explicitly.

`lower_bound()` and `upper_bound()` remain available but are no longer used in
the scalar probability calculation.

### Equation

For `X ~ N(mu,v)` and threshold `b`:

```text
GreaterThan: P(X >= b) = Phi((mu-b)/sqrt(v))
LessThan:    P(X <= b) = Phi((b-mu)/sqrt(v))
output:      [p,p]
```

At zero variance, the result is the deterministic non-strict indicator, so
equality returns one for both orientations.

### Numerical evidence

| Case | Before | After/reference |
|---|---:|---:|
| `N(0,1)`, `X >= 0`, default confidence shift 2 | `[0.022750,0.977250]` | `[0.500000,0.500000]` |
| `N(1,4)`, `X >= 0` | `[0.066807,0.993790]` | `[0.691462,0.691462]` |
| `N(1,4)`, `X <= 0` | `[0.006210,0.933193]` | `[0.308538,0.308538]` |
| Zero variance at equality | `NaN` | `[1,1]` |

Patch 1 reduced strict xfails from 30 to 23. Rectangle, temporal, smoothing,
planner, Monte Carlo, and complexity behavior were not changed.

---

## Patch 2 — Shared Gaussian residual probability

### Production changes

- `src/pdstl/probability.py`: introduced the shared differentiable
  `gaussian_residual_probability(mean, variance)` primitive.
- `src/models/dynamics.py`: `GaussianBelief` delegates to the primitive.
- `src/planning/environment.py`: supported generic planning half-space
  calculations use the same primitive without `variance + 1e-6`.

Rectangle Boolean composition remained unchanged until Patch 3.

### Equation

For `R ~ N(m,v)`:

```text
v > 0: P(R >= 0) = Phi(m/sqrt(v))
v = 0: P(R >= 0) = 1{m >= 0}
v < 0: reject with ValueError
```

### Numerical evidence

For variances `1e-16`, `1e-12`, and `1e-8`, evaluating residual means
`[-sqrt(v),0,sqrt(v)]` gives
`[0.158655,0.500000,0.841345]` in every case. At `m=0.2`, `v=0.7`:

| Quantity | Value |
|---|---:|
| Probability | 0.594464935333 |
| `dP/dm` | 0.463396374918 |
| `dP/dv` | -0.066199482131 |

Patch 2 resolved one additional strict xfail and produced 93 passes with 22
xfails. It did not change rectangles, temporal operators, smoothing, planner
terminology, Monte Carlo, or complexity behavior.

---

## Patch 3 — Generic rectangular Fréchet semantics

### Production changes

Only generic static `RectangularGoalPredicate` and
`RectangularObstaclePredicate` in `src/planning/environment.py` changed. The
generic API remains 2-D. Crazyflie production predicates remain unchanged.

### Equations

For each coordinate:

```text
p_i = P(l_i <= X_i <= u_i)
L_box = max(0, sum_i p_i - (d-1))
U_box = min_i p_i
goal = [L_box,U_box]
obstacle = [1-U_box,1-L_box]
```

No generic marginal probabilities are multiplied and no independence
assumption is made. Zero-variance box boundaries are included.

### Numerical evidence

For `N(0,I)` and `[-1,1]^2`:

| Quantity | Value |
|---|---:|
| One-axis interval probability | 0.682689492137 |
| Generic inside interval | `[0.365378984274,0.682689492137]` |
| Exact independent inside | 0.466064942674 |
| Generic obstacle interval | `[0.317310507863,0.634621015726]` |
| Exact independent outside | 0.533935057326 |

For correlation `rho=0.9`, the seeded SciPy inside probability is
`0.596359949728`, inside the same generic interval. Changing only off-diagonal
covariance does not change the generic marginal bounds. Deterministic points
strictly inside or on any included boundary return goal `[1,1]` and obstacle
`[0,0]`; a strictly outside point returns the complements.

Patch 3 resolved two strict rectangle xfails and produced 103 passes with 20
xfails. Seeded interval invariants and Crazyflie-product containment pass.

---

## Patch 4 — Correct exact bounded StoRI Until

### Production changes

Only the `scale <= 0` path of `Until.robustness_trace` in
`src/pdstl/operators.py` changed. The smooth path and finite-horizon candidate
handling were retained.

### Equation

For each candidate `tau`:

```text
l_phi_min(t,tau) = min_{s=t,...,tau} l_phi(s)
u_phi_min(t,tau) = min_{s=t,...,tau} u_phi(s)
c_down(tau) = max(l_phi_min(t,tau) + l_psi(tau) - 1, 0)
c_up(tau) = min(u_phi_min(t,tau), u_psi(tau))
r_down(t) = max_tau c_down(tau)
r_up(t) = max_tau c_up(tau)
```

### Before/after evidence

For `phi=[0.6,0.6]`, `psi=[0.7,0.7]`, `I=[0,0]`, the former empty-prefix,
pointwise-min path returned `[0.7,0.7]`; exact StoRI returns `[0.3,0.6]`.

For `phi(0)=[0.9,0.9]`, `phi(1)=[0.1,0.1]`, `psi(1)=[0.9,0.9]`, and
`I=[1,1]` at `t=0`, the former exclusive prefix returned `[0.9,0.9]`; the
inclusive result is `[0.0,0.1]`.

Production exact Until matches the direct reference for two nonconstant traces
over `[0,0]`, `[0,1]`, `[0,2]`, `[1,2]`, and the implementation's existing
`[0,infinity]` candidate handling. A fixed-seed check compared 100 valid trace
pairs over all five intervals: 500 full traces and 2,500 output intervals all
matched within `1e-7`. Batch shape, float64 dtype, and device preservation pass.

Patch 4 resolved all six F1–F3 exact-Until xfails. The preserved smooth Until
path no longer converges to corrected exact mode and is recorded as one new
strict G1 xfail. The final result is 115 passes and 15 xfails.

---

## Semantic validation visualizations

### Scope and implementation

`tools/visualize_pdstl_semantics.py` is deterministic, headless by default,
uses production functions for current semantics, and keeps former behavior in
clearly named local helpers. It supports all figures by default and selected
runs such as:

```text
python tools/visualize_pdstl_semantics.py
python tools/visualize_pdstl_semantics.py --figure atomic
python tools/visualize_pdstl_semantics.py --figure rectangle
python tools/visualize_pdstl_semantics.py --figure until
python tools/visualize_pdstl_semantics.py --figure smoothing
```

Every plotted correction is checked against a direct mathematical reference.
Every PDF and PNG is asserted to exist and be nonempty after export. No
notebook state is required.

### Figure 1 — Gaussian atomic correction

- **Demonstrates:** the current direct atom `Phi(m)` versus the former
  confidence-shifted interval, plus deterministic zero-variance equality.
- **Example:** `R ~ N(m,1)`, former `k=2`, and zero-variance
  `1{m >= 0}`.
- **Key values:** at `m=0`, current `[0.5,0.5]`, former
  `[0.022750,0.977250]`; zero-variance equality is one.
- **Files:**
  [PDF](figures/semantic_validation/01_atomic_probability.pdf),
  [PNG](figures/semantic_validation/01_atomic_probability.png).
- **Classification:** supports Patches 1–2.

### Figure 2 — Rectangle dependence robustness

- **Demonstrates:** the true correlated Gaussian box probability stays inside
  the generic Fréchet interval across `rho in [-0.95,0.95]`; the former product
  is exact only at independence. The second panel shows the obstacle complement.
- **Example:** zero-mean unit marginals and box `[-1,1]^2`.
- **Key values:** generic inside interval
  `[0.365378984274,0.682689492137]`; true sweep range
  `[0.466064942674,0.621639025943]`; at `rho=0.9`, `0.596359949728`;
  maximum deviation from the product is `0.155574083269`. The obstacle interval
  is `[0.317310507863,0.634621015726]`.
- **Files:**
  [PDF](figures/semantic_validation/02_rectangle_frechet.pdf),
  [PNG](figures/semantic_validation/02_rectangle_frechet.png).
- **Classification:** supports Patch 3.

### Figure 3 — Exact bounded Until correction

- **Demonstrates:** the effect of the inclusive prefix and Fréchet lower
  candidate on both interval endpoints.
- **Example:** four-step nonconstant phi/psi traces and `I=[0,2]`.
- **Key values:** corrected output is
  `[[0.1,0.3],[0.0,0.3],[0.3,0.7],[0.3,0.7]]`; former output is
  `[[0.9,0.9],[0.9,0.95],[0.8,0.9],[0.8,0.9]]`; maximum difference is `0.90`.
- **Files:**
  [PDF](figures/semantic_validation/03_until_correction.pdf),
  [PNG](figures/semantic_validation/03_until_correction.png).
- **Classification:** supports Patch 4.

### Figure 4 — Semantic-audit progress

- **Demonstrates:** remaining known strict expected failures after each
  validated stage, without implying fixed test totals or percentage correctness.
- **Example/key values:** `30 -> 23 -> 22 -> 20 -> 15` from initial audit
  through Patch 4.
- **Files:**
  [PDF](figures/semantic_validation/04_audit_progress.pdf),
  [PNG](figures/semantic_validation/04_audit_progress.png).
- **Classification:** engineering/research progress evidence.

### Figure 5 — Current smooth temporal extrema

- **Demonstrates:** unnormalized production `Minish`/`Maxish` convergence and
  their excursions outside `[0,1]` at finite beta.
- **Examples:** `[0.05,0.05,0.05]`, `[0.20,0.35,0.80]`, and
  `[0.99,0.99,0.99]`; beta `1,2,5,10,20,50,100,500`.
- **Key values:** `smin_10([0.05]*3)=-0.059861228867` and
  `smax_10([0.99]*3)=1.099861228867`.
- **Files:**
  [PDF](figures/semantic_validation/05_smoothing_extrema.pdf),
  [PNG](figures/semantic_validation/05_smoothing_extrema.png).
- **Classification:** exposes the remaining bounded G2 defect.

### Figure 6 — Smooth-min gradient distribution

- **Demonstrates:** gradient mass is distributed at small beta and concentrates
  around the active minimum as beta grows; the hard minimum selects one element.
- **Example:** `[0.20,0.21,0.60,0.90]`, beta `1,5,10,50,100`.
- **Key values:** beta 1 gradient is
  `[0.316761,0.313609,0.212331,0.157299]`; beta 100 is approximately
  `[0.731059,0.268941,0,0]`; exact is `[1,0,0,0]`.
- **Files:**
  [PDF](figures/semantic_validation/06_smoothing_gradients.pdf),
  [PNG](figures/semantic_validation/06_smoothing_gradients.png).
- **Classification:** characterizes why bounded smoothing is useful.

### Figure 7 — Current Boolean Fréchet dead region

- **Demonstrates:** `max(l1+l2-1,0)` and its zero-gradient region below
  `l1+l2=1`.
- **Example:** a `101 x 101` grid over `[0,1]^2`.
- **Key values:** output and gradient are exactly zero throughout
  `l1+l2<1`; requesting scale 50 produces exactly the same output and gradient
  as exact scale `-1`.
- **Files:**
  [PDF](figures/semantic_validation/07_and_gradient_region.pdf),
  [PNG](figures/semantic_validation/07_and_gradient_region.png).
- **Classification:** exposes the remaining bounded G3–G4 design issue.

### Figure 8 — Smooth Until mismatch after Patch 4

- **Demonstrates:** current smooth Until does not converge to corrected exact
  bounded Until.
- **Example:** the same four-step trace and `I=[0,2]` used in Figure 3;
  beta `1,2,5,10,20,50,100,500`.
- **Key values:** infinity-norm errors are approximately
  `[0.908491,0.797962,0.815286,0.868763,0.893654,0.899866,0.900000,0.900000]`.
- **Files:**
  [PDF](figures/semantic_validation/08_smooth_until_mismatch.pdf),
  [PNG](figures/semantic_validation/08_smooth_until_mismatch.png).
- **Classification:** exposes the remaining bounded G1 defect.

### What the visualizations imply for the next smooth-semantics design

The bounded differentiable layer must be evaluated against four simultaneous
requirements: convergence to the corrected exact semantics, preservation of
valid `[0,1]` intervals, useful gradients away from active hard extrema, and a
smooth Until construction aligned with the inclusive Fréchet recursion. The
figures deliberately characterize these constraints without selecting or
implementing a replacement smoothing equation.

Explicit or delayed unbounded temporal intervals are outside the bounded
finite-time paper scope used for these figures. Their existing audit findings
remain recorded but are not Direction-A blockers for this bounded study.

## Current remaining findings

| Finding | Severity | Status/scope | Responsible location |
|---|---|---|---|
| Planning predicates exceed declared `Belief` API | High | 4 strict xfails | `src/pdstl/base.py`, `src/planning/environment.py` |
| `TorchGaussianBelief` cannot serve core scalar atoms | High | 2 strict xfails | `src/planning/planner.py:TorchGaussianBelief` |
| Explicit `[0,infinity]` Always/Eventually conversion | Medium | 2 strict xfails; outside bounded paper scope | `src/pdstl/operators.py:Always/Eventually` |
| Smooth Until does not approach corrected exact Until | High | 1 strict xfail; bounded smoothing issue | `src/pdstl/operators.py:Until.robustness_trace` |
| Smooth extrema can leave `[0,1]` | High | 3 strict xfails; bounded smoothing issue | `src/pdstl/operators.py:Minish/Maxish` |
| Boolean operators ignore smoothing scale | High | Characterized | `src/pdstl/operators.py:And/Or` |
| Planner clamp has zero-gradient outer regions | High | Characterized | `src/planning/planner.py:Planner._compute_loss` |
| Planner StoRI endpoint terminology is misleading | Low | Characterized | planner/runners/config labels |
| Monte Carlo ignores `x0_cov` sampling | High | 1 strict xfail | `src/evaluation/monte_carlo.py:rollout_controls` |
| Empirical metric differs from actual timed STL formula | High | 1 strict xfail | `src/evaluation/metrics.py:evaluate_rollouts` |
| Empirical satisfaction lacks confidence interval | Medium | 1 strict xfail | `src/evaluation/metrics.py:evaluate_rollouts` |
| Dense temporal implementation exceeds general O(T) claim | High | Static finding | `src/pdstl/operators.py`, complexity benchmark |

## Final validation

| Command | Result |
|---|---:|
| `python tools/visualize_pdstl_semantics.py` | 8 figures, 16 nonempty files |
| `pytest -q -rxX tests/audit_pdstl` | 115 passed, 15 xfailed |
| `pytest -q tests/test_pdstl_operators.py` | 3 passed |
| `pytest -q experiments/crazyflie/tests/test_figure8_planning.py experiments/crazyflie/tests/test_pipeline.py` | 50 passed |
| `git diff --check` | Passed |

The audit and existing suites contain 183 pytest node IDs in total. Final runs
had no ordinary failures, unexpected passes, skips, or blocked tests. All 16
figure files were visually inspected and are nonempty.
