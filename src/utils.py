"""Configuration and device helpers shared by the scenario runners."""

import os
from contextlib import contextmanager

import torch
import yaml


def get_device():
    """Use the requested device, then CUDA if available, then CPU."""
    requested = os.environ.get("PDSTL_DEVICE")
    if requested:
        return torch.device(requested)
    if (
        torch.cuda.is_available()
        and os.environ.get("PDSTL_USE_CUDA", "1") != "0"
    ):
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path):
    """Load YAML, resolving relative paths from the project root."""
    if not os.path.isabs(path):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, path)
    with open(path) as stream:
        return yaml.safe_load(stream)


class _SkippedRun(Exception):
    """Leave a disabled project block without executing its body."""


@contextmanager
def skip_run(flag, label):
    """Keep the project-by-project ``with skip_run(...), check()`` entry point."""
    if flag not in ("run", "skip"):
        raise ValueError("run flag must be 'run' or 'skip'")

    @contextmanager
    def check():
        if flag == "skip":
            print(f"Skipping {label}", flush=True)
            raise _SkippedRun
        print(f"Running {label}", flush=True)
        yield

    try:
        yield check
    except _SkippedRun:
        pass
