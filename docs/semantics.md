# Core semantic contract

This document specifies the equations implemented on `pdstl-core`. They operate on probability intervals supplied by an upstream predictor and predicate evaluator. Their temporal meaning is pointwise probabilistic robustness. They do not compute the probability that one stochastic realization satisfies the entire specification.

## Inputs and Boolean equations

At each batch element and prediction step, an atom supplies an interval `[L,U]` containing its event probability. Numeric validity is checked; mathematical coverage is the provider's responsibility. Correlations are not inferred or replaced with an independence assumption.

For child scores `[L1,U1]` and `[L2,U2]`, direct evaluation uses:

| Operator | Lower endpoint | Upper endpoint |
|---|---|---|
| Not | `1-U1` | `1-L1` |
| And | `max(0,L1+L2-1)` | `min(U1,U2)` |
| Or | `max(L1,L2)` | `min(1,U1+U2)` |
| Implies | Evaluate `Or(Not(left),right)` | Same composition |

For combinations of events at one prediction step, these are probability bounds. For general temporal child formulas, they define the chosen compositional score; that does not turn those children into trajectory event probabilities. Repeated or logically dependent predicates can yield loose intervals because the evaluator does not simplify logical dependencies. Equal atomic endpoints therefore need not give equal endpoints after Boolean composition.

## Temporal equations

All intervals refer to discrete prediction indices, include both endpoints, and are relative to each evaluation origin `t`.

Always takes the minimum of each child endpoint over `t+a,...,t+b`. Eventually takes the maximum of each child endpoint over the same window.

For Until, write `PL(t,tau)` and `PU(t,tau)` for the minimum of the respective left-child endpoints over the **inclusive** prefix `t,...,tau`. For witnesses `tau=t+a,...,t+b`:

```text
L_Until(t) = max_tau max(0, PL(t,tau) + L_right(tau) - 1)
U_Until(t) = max_tau min(PU(t,tau), U_right(tau))
```

In particular, a witness at `tau=t` includes the left child's value at `t`; it does not use a vacuous prefix of one. This is the selected StoRI-style convention and changes the old implementation's exclusive prefix. A downstream deterministic SSR evaluator must explicitly document which Until convention it uses.

These equations preserve `0 <= L <= U <= 1` on valid inputs. They also preserve interval refinement: increasing atomic lower endpoints and decreasing atomic upper endpoints cannot widen the resulting direct interval. This follows by structural induction: complement swaps endpoints, and each other endpoint expression is monotone in its corresponding child endpoints. These properties concern the defined score, not the coverage of a trajectory-level probability.

## Lookahead and complete outputs

The required lookahead `h` is:

| Formula | Lookahead |
|---|---|
| Atom | `0` |
| Not | Child lookahead |
| And, Or, Implies | Maximum child lookahead |
| Always, Eventually on `[a,b]` | `b + child lookahead` |
| Inclusive Until on `[a,b]` | `b + max(left lookahead, right lookahead)` |

With `T` predictions indexed `0,...,T-1`, only origins `0,...,T-h-1` are returned. Binary children with different lookaheads are aligned at those common origins. A formula requiring more future than is supplied raises an error. There is no terminal padding, partial-window estimate, or infinite-horizon interpretation.

Always and Eventually reverse the child trace, shift each sample into a register through matrix `M` and insertion vector `b`, and reverse the output trace back. Internal register placeholders never enter a returned reduction: the evaluator waits for a full register. At that point the first `b-a+1` register positions contain the desired future window in reverse order, which leaves min/max unchanged. Matrices are registered module buffers; arithmetic follows the input dtype and device.

Each call evaluates a supplied prediction from scratch. The recurrence is internal to that call; it does not reuse predicted values across MPC updates. Until independently reuses left-prefix reductions while enumerating witnesses.

For a unary window with register size `R=b+1`, the retained dense matrix implementation costs `O(B*T*R^2)` arithmetic and `O(B*R+R^2)` live register/matrix storage, excluding outputs and autograd storage. Until's reused prefixes cost `O(B*T*(b+1))` arithmetic per node. Full formula cost is the sum over evaluated nodes; shared subexpressions are not memoized. No general linear-time claim is made.

## Existing smoothing convention

For positive `scale = beta`, every min/max reduction is replaced by unnormalized log-sum-exp:

```text
smoothmax(x) = log(sum_i exp(beta*x_i)) / beta
smoothmin(x) = -log(sum_i exp(-beta*x_i)) / beta
```

The implementation uses `torch.logsumexp` for numerical stability. For `n` terms, the absolute error relative to the exact reduction is at most `log(n)/beta`; smoothmax is above max and smoothmin is below min. For a composed formula, those local directions do not imply that a smooth lower endpoint is conservative. In particular, smooth Boolean clamp operations can produce crossed endpoints.

Let `E(phi)` bound the maximum absolute endpoint error across every returned origin on the same input trace. `smoothing_error(beta)` uses these sufficient bounds:

| Formula | Error bound |
|---|---|
| Supplied atom | `0` |
| Not | `E(child)` |
| And, Or, Implies | `E(left)+E(right)+log(2)/beta` |
| Always, Eventually | `E(child)+log(b-a+1)/beta` |
| Until | `E(left)+E(right)+(log(b+1)+log(2)+log(b-a+1))/beta` |

The proof uses the reduction error above and the fact that min/max and their smooth versions are 1-Lipschitz in the maximum norm; additions can sum the two child errors. Until's prefix recurrence is associative log-sum-exp over at most `b+1` samples, so the prefix contributes `log(b+1)/beta`. These bounds ignore floating-point roundoff and may be loose. Custom formula implementations must declare their own bound; the base class raises rather than assuming zero error.

Direct evaluation (`scale <= 0`) has zero smoothing error. Positive-scale evaluation remains an optimization approximation, even when its endpoints happen to lie in `[0,1]`. Reporting uses the direct equations on the optimized prediction. A control objective's gradient reaches the controls only if the upstream prediction and probability provider preserve autograd; converting through NumPy breaks that chain.

## Boundaries of this revision

The core supplies an evaluator and differentiable approximation, not a complete stochastic MPC algorithm. Gaussian parameter ambiguity, tube construction, joint predicate integration, sampled trajectory SSR, online monitoring, planner constraints, recursive feasibility, stability, and scenario validation require their own modeling and implementation. Legacy comparison adapters remain available for migration and do not establish these properties.
