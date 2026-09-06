import os
import sys
from contextlib import contextmanager

import numpy as np
import torch
import yaml


def get_device():
    """Return the configured torch device.

    This project defaults to CPU because some lab machines expose CUDA libraries
    even when the installed GPU/driver pair cannot actually initialize. Set
    PDSTL_DEVICE=cuda or PDSTL_USE_CUDA=1 to opt into CUDA explicitly.
    """
    requested = os.environ.get("PDSTL_DEVICE")
    if requested:
        return torch.device(requested)
    if os.environ.get("PDSTL_USE_CUDA") == "1" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path):
    """Load a YAML config file and return it as a dict.

    Parameters
    ----------
    path : str
        Path to the YAML file. Can be relative to the project root

    """
    if not os.path.isabs(path):
        # Resolve relative paths from the project root (two levels above this file)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


class SkipWith(Exception):
    pass


@contextmanager
def skip_run(flag, f):
    """To skip a block of code.

    Parameters
    ----------
    flag : str
        skip or run.

    Returns
    -------
    None
    """

    @contextmanager
    def check_active():
        deactivated = ["skip"]
        p = ColorPrint()  # printing options
        if flag in deactivated:
            p.print_skip("{:>12}  {:>2}  {:>12}".format("Skipping the block", "|", f))
            raise SkipWith()
        else:
            p.print_run("{:>12}  {:>3}  {:>12}".format("Running the block", "|", f))
            yield

    try:
        yield check_active
    except SkipWith:
        pass


class ColorPrint:
    @staticmethod
    def print_skip(message, end="\n"):
        sys.stderr.write("\x1b[88m" + message.strip() + "\x1b[0m" + end)
        sys.stderr.flush()

    @staticmethod
    def print_run(message, end="\n"):
        sys.stdout.write("\x1b[1;32m" + message.strip() + "\x1b[0m" + end)
        sys.stdout.flush()


def to_steps(interval_sec, t):
    """Convert a time interval [a, b] in seconds to integer step indices.

    Parameters
    ----------
    interval_sec : list of two numbers
        Interval bounds in seconds. The second bound may be np.inf.
    t : array-like
        Time vector (uniformly spaced).

    Returns
    -------
    [a_step, b_step] : list
    """
    dt = float(t[1] - t[0])
    a = int(round(interval_sec[0] / dt))
    b = np.inf if np.isinf(interval_sec[1]) else int(round(interval_sec[1] / dt))
    return [a, b]


def create_belief_trajectory(
    mean_trace, var_trace, confidence_level=1.0, dtype=None, device=None
):
    """Wrap mean/variance arrays into a BeliefTrajectory of GaussianBeliefs.

    Each element holds the belief at one prediction step, shaped [batch, dim]
    per the convention in pdstl.base.

    Parameters
    ----------
    mean_trace : array-like, shape (T,)
    var_trace  : array-like, shape (T,)
    confidence_level : float
        Mean-ambiguity radius in standard deviations; see GaussianBelief.
    dtype : torch.dtype, optional
        Defaults to torch.float32.
    device : torch.device, optional
        Defaults to get_device().

    Returns
    -------
    BeliefTrajectory
    """
    from models.dynamics import GaussianBelief
    from pdstl.base import BeliefTrajectory

    dtype = torch.float32 if dtype is None else dtype
    device = get_device() if device is None else device

    mean = torch.as_tensor(mean_trace, dtype=dtype, device=device).reshape(1, -1, 1)
    var = torch.as_tensor(var_trace, dtype=dtype, device=device).reshape(1, -1, 1)
    beliefs = [
        GaussianBelief(mean[:, i, :], var[:, i, :], confidence_level=confidence_level)
        for i in range(len(mean_trace))
    ]
    return BeliefTrajectory(beliefs)
