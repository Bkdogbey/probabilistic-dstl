"""Two fixed probability intervals for the Boolean-operator experiment."""

import torch


def boolean_example(dtype=torch.float64):
    """Return (bounds_a, bounds_b), each Tensor[1, 1, 2]: A=[0.60,0.90], B=[0.70,0.95]."""
    bounds_a = torch.tensor([[[0.60, 0.90]]], dtype=dtype)
    bounds_b = torch.tensor([[[0.70, 0.95]]], dtype=dtype)
    return bounds_a, bounds_b
