"""Input contract for pdSTL: what one prediction step supplies, and who evaluates it.

Pipeline position
-----------------
    stochastic prediction -> predicate probability intervals -> Boolean/temporal
    evaluation -> lower-bound objective -> control optimization

This module covers the first two stages only. The division of labour is:

    * a **formula** states the requirement (which event it is about),
    * a **belief** evaluates that requirement under its own uncertainty model.

An atomic predicate therefore never inspects a belief's internals, and a belief
never needs to know which temporal operator it sits under.

Shape and endpoint conventions
------------------------------
B is the batch dimension, T the trajectory length and D the **state** dimension.
D is never reinterpreted as a predicate index.

    one input element                    the uncertainty over the state at ONE step
    Belief.value()                       [B, D]     point estimate (optional)
    Belief.probability_bounds(pred)      [B, 2]     [..., 0] lower <= [..., 1] upper
    Predicate.robustness_trace(traj)     [B, T, 2]  stacked over time

The last axis keeps the project's existing [lower, upper] convention. Exact
probabilities are expressed as equal endpoints; non-singleton intervals are the
normal case, not an exception.

Providers must build results from their own tensors so that dtype, device and
autograd all flow through. Constructing a fresh torch.tensor(...) from a Python
float on the differentiable path detaches the gradient; derive dtype and device
from the incoming tensors instead.
"""

from abc import ABC, abstractmethod

import torch

# Endpoints may fall this far outside [0, 1] before validation complains, so
# ordinary floating-point round-off in a CDF evaluation is not reported as
# malformed input.
_BOUNDS_ATOL = 1e-6


class Belief(ABC):
    """The uncertainty over the state at a single prediction step.

    Subclasses need not be Gaussian, and need not expose a state at all: the
    only requirement is the ability to answer, for a predicate, what the lower
    and upper probability of that predicate's event are at this step.
    """

    @abstractmethod
    def probability_bounds(self, predicate):
        """Lower and upper probability of the predicate's event at this step.

        Parameters
        ----------
        predicate : Predicate
            The formula stating the requirement. Implementations read whatever
            they support from it -- `name` for an identified event, or
            `dim`/`threshold`/`sense` for a comparison -- and evaluate it under
            their own uncertainty model.

        Returns
        -------
        Tensor of shape [B, 2], [..., 0] lower <= [..., 1] upper, both in
        [0, 1], carrying this belief's dtype and device and preserving autograd.

        Raises
        ------
        ValueError
            If this uncertainty model cannot evaluate the given predicate. The
            message must name the predicate, so an unsupported event fails
            loudly instead of being served the wrong number.
        """

    def value(self):
        """Optional point estimate of the state, [B, D].

        Not part of the predicate contract -- no operator calls it. It exists
        for consumers that legitimately need a representative state, such as the
        deterministic STL baseline. Beliefs that supply probabilities directly
        and hold no state need not implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a point estimate via value()."
        )


class BeliefTrajectory:
    """An ordered container of per-step beliefs.

    It organizes steps and nothing else: it computes no probabilities and holds
    no formula state. Trace assembly lives in pdstl.operators.Predicate.
    """

    def __init__(self, beliefs):
        self.beliefs = list(beliefs)

    def __getitem__(self, t):
        return self.beliefs[t]

    def __len__(self):
        return len(self.beliefs)

    def __iter__(self):
        return iter(self.beliefs)

    def suffix(self, t):
        """The trajectory from step t onwards, as the same container type."""
        return type(self)(self.beliefs[t:])


class OnlineBeliefTrajectory(BeliefTrajectory):
    """An appendable input container for streaming predictions.

    This is **not** an incremental temporal monitor. append() adds an input
    element; it does not advance, retain or invalidate any temporal-operator
    state, and re-evaluating a formula recomputes the whole trace from the
    beginning. An incremental monitor would have to live with the recurrent
    cells in pdstl.operators and does not exist yet.
    """

    def __init__(self, beliefs=None):
        super().__init__([] if beliefs is None else beliefs)

    def append(self, belief):
        self.beliefs.append(belief)

    @classmethod
    def from_list(cls, lst):
        return cls(lst)


class ProbabilityBelief(Belief):
    """A step whose predicate probability bounds are supplied already computed.

    For callers that do not expose a belief representation but do have interval
    probabilities per event, for instance from a learned predictor or an
    upstream verification tool. Events are identified by the predicate's name,
    so bounds registered for one event can never be served for another.

    Validation of the supplied numbers is limited to well-formedness (finite,
    ordered, within [0, 1]). Nothing here can verify that an interval actually
    *covers* the true event probability -- that claim rests with whoever
    produced the numbers.

    Parameters
    ----------
    bounds : dict of str -> Tensor
        Event name to a [B, 2] tensor, lower endpoint first. Tensors are taken
        as-is, so one that requires grad keeps its graph.
    value : Tensor, optional
        Point estimate [B, D], if the caller happens to have one.
    """

    def __init__(self, bounds, value=None):
        self.bounds = {}
        for name, b in bounds.items():
            b = torch.as_tensor(b)
            if b.ndim != 2 or b.shape[-1] != 2:
                raise ValueError(
                    f"ProbabilityBelief: bounds for event {name!r} must have shape "
                    f"[batch, 2], got {tuple(b.shape)}."
                )
            self.bounds[name] = b
        self._value = value

    def probability_bounds(self, predicate):
        name = getattr(predicate, "name", None)
        if name is None:
            raise ValueError(
                f"ProbabilityBelief serves events by name, but {predicate!r} is "
                f"unnamed. Give the predicate a name matching one of: "
                f"{sorted(self.bounds)}."
            )
        try:
            return self.bounds[name]
        except KeyError:
            raise ValueError(
                f"ProbabilityBelief has no bounds for event {name!r}. "
                f"Known events: {sorted(self.bounds)}."
            ) from None

    def value(self):
        if self._value is None:
            return super().value()
        return self._value


def _first_bad_step(mask):
    """Lowest time index at which mask ([B, T] bool) is set, or -1."""
    steps = mask.any(dim=0).nonzero()
    return int(steps[0].item()) if steps.numel() else -1


def check_probability_bounds(trace, predicate=None):
    """Reject malformed probability bounds before they reach the operators.

    Checks well-formedness only: finite, lower <= upper, and both endpoints
    within [0, 1] up to _BOUNDS_ATOL. It raises rather than clamping, because
    clamping would silently alter the value *and* its gradient.

    Numeric validity is not evidence of probability coverage: a well-formed
    interval can still fail to contain the true event probability. That
    guarantee belongs to the provider.

    Parameters
    ----------
    trace : Tensor
        [B, T, 2] stacked atom output.
    predicate : Predicate, optional
        Used only to name the offending atom in error messages.

    Raises
    ------
    ValueError
        Naming the predicate and the first offending time step.
    """
    who = "predicate"
    if predicate is not None:
        label = getattr(predicate, "name", None) or type(predicate).__name__
        who = f"predicate {label!r}"

    if trace.ndim != 3 or trace.shape[-1] != 2:
        raise ValueError(
            f"{who} produced shape {tuple(trace.shape)}; expected [batch, time, 2]."
        )

    finite = torch.isfinite(trace)
    if not bool(finite.all()):
        t = _first_bad_step(~finite.all(dim=-1))
        raise ValueError(f"{who} produced non-finite probability bounds at step {t}.")

    lower, upper = trace[..., 0], trace[..., 1]

    unordered = lower > upper + _BOUNDS_ATOL
    if bool(unordered.any()):
        t = _first_bad_step(unordered)
        raise ValueError(
            f"{who} produced lower > upper at step {t} "
            f"(lower={lower[:, t].max().item():.6g}, "
            f"upper={upper[:, t].min().item():.6g})."
        )

    out_of_range = (trace < -_BOUNDS_ATOL) | (trace > 1.0 + _BOUNDS_ATOL)
    if bool(out_of_range.any()):
        t = _first_bad_step(out_of_range.any(dim=-1))
        raise ValueError(
            f"{who} produced probability bounds outside [0, 1] at step {t} "
            f"(min={trace[:, t].min().item():.6g}, max={trace[:, t].max().item():.6g})."
        )
