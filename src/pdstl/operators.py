"""pdSTL formula core: atomic predicates and pointwise Boolean operators.

Every Formula is a torch.nn.Module. forward(source) queries a
ProbabilitySource and returns Tensor[B, T, 2] with [..., 0] = lower and
[..., 1] = upper. Boolean combinations use dependence-agnostic Frechet
bounds; no independence assumption, no smoothing.
"""

import torch

__all__ = ["And", "Formula", "Not", "Or", "Predicate"]


class Formula(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, source):
        raise NotImplementedError

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


class Predicate(Formula):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, source):
        return torch.stack([source.bounds(self, t) for t in range(len(source))], dim=1)

    def __str__(self):
        return self.name


class Not(Formula):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def forward(self, source):
        bounds = self.child(source)
        lower_not = 1 - bounds[..., 1]
        upper_not = 1 - bounds[..., 0]
        return torch.stack([lower_not, upper_not], dim=-1)

    def __str__(self):
        return f"¬({self.child})"


class And(Formula):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, source):
        left_bounds = self.left(source)
        if self.left is self.right:
            return left_bounds
        right_bounds = self.right(source)

        left_lower, left_upper = left_bounds[..., 0], left_bounds[..., 1]
        right_lower, right_upper = right_bounds[..., 0], right_bounds[..., 1]

        lower = torch.clamp(left_lower + right_lower - 1, min=0.0)
        upper = torch.minimum(left_upper, right_upper)
        return torch.stack([lower, upper], dim=-1)

    def __str__(self):
        return f"({self.left}) ∧ ({self.right})"


class Or(Formula):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, source):
        left_bounds = self.left(source)
        if self.left is self.right:
            return left_bounds
        right_bounds = self.right(source)

        left_lower, left_upper = left_bounds[..., 0], left_bounds[..., 1]
        right_lower, right_upper = right_bounds[..., 0], right_bounds[..., 1]

        lower = torch.maximum(left_lower, right_lower)
        upper = torch.clamp(left_upper + right_upper, max=1.0)
        return torch.stack([lower, upper], dim=-1)

    def __str__(self):
        return f"({self.left}) ∨ ({self.right})"
