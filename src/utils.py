import os
import sys
from contextlib import contextmanager

import numpy as np
import torch
import yaml


def get_device():
    """Use the requested device, otherwise prefer CUDA when available, then CPU."""
    requested = os.environ.get("PDSTL_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available() and os.environ.get("PDSTL_USE_CUDA", "1") != "0":
        return torch.device("cuda")
    return torch.device("cpu")
# to run gpu: PDSTL_DEVICE=cuda python src/main.py

def load_config(path):
    """Load a YAML file; relative paths resolve from the project root."""
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
    """Context manager that runs or skips a block: flag is 'run' or 'skip'."""

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
        sys.stderr.write("\x1b[33m" + message.strip() + "\x1b[0m" + end)
        sys.stderr.flush()

    @staticmethod
    def print_run(message, end="\n"):
        sys.stdout.write("\x1b[1;32m" + message.strip() + "\x1b[0m" + end)
        sys.stdout.flush()


def to_steps(interval_sec, t):
    """Map an interval in seconds to the nearest indices of a uniform time grid [a, b]; b may be inf."""
    t = np.asarray(t)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("time vector must be one-dimensional with at least two samples")
    differences = np.diff(t)
    if not np.all(differences > 0):
        raise ValueError("time vector must be strictly increasing")
    if not np.allclose(differences, differences[0]):
        raise ValueError("time vector must be uniformly spaced")

    dt = float(differences[0])
    a = int(round(interval_sec[0] / dt))
    b = np.inf if np.isinf(interval_sec[1]) else int(round(interval_sec[1] / dt))
    return [a, b]
