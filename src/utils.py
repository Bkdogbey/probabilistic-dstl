import os
import sys
from contextlib import contextmanager

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
