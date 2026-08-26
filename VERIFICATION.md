# pdSTL verification suite

**These are verification examples, not the final application experiments.**

They exist to show that the implemented probability-first pdSTL pipeline behaves as intended —
before moving on to lane-change, Crazyflie, Sawyer, or any other RA-L application. Every bound
reported anywhere below comes from the **exact hard probability semantics**: no smooth surrogate,
no `beta`, no softplus/logsumexp, no margin objective, no chance constraint, no risk allocation,
and no independence assumption.

## Running it

```bash
python src/main.py
```

That is the whole pipeline. It prints a numerical table per block and writes every figure to
`figures/verification/` as both PDF and PNG. Individual blocks are toggled by flipping
`"run"` / `"skip"` in the `skip_run(...)` calls in [src/main.py](src/main.py); scenario parameters
live in [configs/stl_demos.yaml](configs/stl_demos.yaml) and can be changed without touching code.

| where | what |
|---|---|
| [src/main.py](src/main.py) | entry point — the toggle blocks |
| [src/verification.py](src/verification.py) | the scenarios, evaluation, and printed tables |
| [src/visualization/verification.py](src/visualization/verification.py) | the figures |
| [src/models/dynamics.py](src/models/dynamics.py) | belief propagation (`linear_system`, `linear_gaussian_rollout`) |
| [tests/test_pdstl_verification.py](tests/test_pdstl_verification.py) | automated checks of the numerical core |

Every formula is evaluated on **all three backends** — the reference interpreter
(`propagate.evaluate`), the compiled graph (`compile_formula`), and the recurrent evaluator
(`compile_recurrent_formula`) — against bit-identical atomic inputs, and agreement is *asserted*,
not assumed. The reference interpreter reads the same tensors as the other two through
`pdstl.TensorProbabilitySource`.

---

## Verification A — `F[5,10](x >= 8)`

**Demonstrates the temporal union.**

A scalar Gaussian trace `X_k ~ N(mu_k, sigma_k^2)` whose mean climbs toward the threshold. Before
the window `x >= 8` is very unlikely; inside `[5, 10]` the atomic probabilities rise visibly. The
satisfaction event is the *union* of the per-time events, so

```
L_F = max_k p_k = 0.342043     (attained at k = 10)
U_F = min(1, sum_k p_k) = 0.718498
```

The mean approaches the threshold rather than crossing it, which keeps `sum_k p_k` below 1 so the
upper bound stays informative instead of saturating at exactly 1.

Figure: `figures/verification/eventually.pdf` — the state trace with its `±2σ` band above, the
atomic probabilities and the resulting interval below.

## Verification B — `G[2,6]((z > 3) AND (z < 5))`

**Demonstrates probabilistic conjunction followed by temporal intersection.**

The band is represented *logically*, as two halfspace predicates on the same random variable, so
pdSTL resolves the conjunction with its own Fréchet rules. Result: `[0.360806, 0.747507]`.

Two things this pins down:

- **The conjunction is exact.** For two halfspaces on one scalar Gaussian,
  `max(0, p_low + p_high - 1)` *is* `Phi((5-mu)/s) - Phi((3-mu)/s)`. Measured agreement with the
  analytic band probability: **2.2e-16**. The analytic value is plotted as an external reference
  only — it is never supplied to pdSTL, which sees nothing but the two event probabilities.
- **It is not a product.** `p_low * p_high` would assume independence between two events defined on
  the *same* random variable. At the widened-uncertainty step the product gives 0.6669 against a
  true 0.6397 — an overestimate of 0.027, which would be unsound as a lower bound.

The temporal intersection then uses the **post-identity-reduction** operand count `m` (here 5, since
the five window events are all distinct):
`L_G = max(0, sum_k L_k - (m-1))`, `U_G = min_k U_k`.

Figure: `figures/verification/always.pdf` — state trace with the admissible band; the two event
probabilities, the pdSTL band, the analytic reference and the rejected product; then the temporal
aggregation.

## Verification C — `G[0,12] Safe AND F[6,12] Goal`

**Demonstrates the complete pipeline and direct hard-bound optimization.**

A 2-D linear stochastic system `x_{k+1} = A x_k + B u_k + w_k`, `w_k ~ N(0, Q)`, with open-loop
controls, so `mu` and `Sigma` propagate exactly. Safe and Goal are rectangles, each expressed as the
**conjunction of its four halfspaces** — no bespoke rectangle probability, and no independence
assumed between faces.

> Linear-Gaussian dynamics are used here only because they give transparent, hand-checkable
> mean/covariance propagation. **pdSTL requires none of it** — not linearity, not Gaussian noise,
> not mean/covariance beliefs. It consumes atomic event probabilities from whatever provider
> produces them.

**C1 — forward.** With a fixed hand-designed control sequence:

| formula | lower | upper |
|---|---|---|
| `G Safe` | 0.952244 | 0.991928 |
| `F Goal` | 0.351085 | 0.608724 |
| `G Safe AND F Goal` | 0.303328 | 0.608724 |

`G Safe` is visibly Fréchet-active (0.952, not 1.0), and the conjunction sits on an active branch
(`L_G + L_F - 1 = +0.303 > 0`).

**C2 — optimization.** The same system and formula, with the controls made optimizable. The
objective is `-P_lower(phi)`, where `P_lower` is the **hard pdSTL lower probability bound itself** —
not a robustness value and not a surrogate. Adam (lr 0.05, 120 iterations):

```
P_lower:  0.303328 -> 0.808093
P_upper:  0.608724 -> 0.991928
gradient: 1.39e+00 -> 1.12e-03
```

Controls are box-constrained by a `tanh` parameterization. That squashing is an
optimizer/application choice and forms no part of the pdSTL semantics.

**C3 — the known zero-gradient plateau.** Same system with the goal moved out of reach:

```
L_G = 0.952244,  L_F = 0.000000
Frechet pre-clamp  L_G + L_F - 1 = -0.047756  < 0
=> P_lower = 0.000000,  ||grad P_lower|| = 0.000000e+00
```

This is the expected behavior of the exact hard semantics and is **reported, not repaired**. It is
what separates the claim being made here — *the end-to-end differentiable pipeline works* — from a
claim that is **not** being made: that the hard lower semantics is globally gradient-rich. It is
not. No smoothing, margin, or surrogate is introduced to hide it.

Figures: `figures/verification/stochastic_forward.pdf` and
`figures/verification/stochastic_optimization.pdf`.

---

## Tests

```bash
python -m pytest tests/test_pdstl_verification.py -q
```

The tests check the numbers, never the rendered images, and import the same builders `main.py`
calls — so what is asserted is exactly what the figures show. They also pin
`linear_gaussian_rollout` to the pre-existing `differentiable_rollout` for `A = I, B = dt·I`, so the
generalized propagator cannot silently drift from the one the rest of the suite already relies on.
