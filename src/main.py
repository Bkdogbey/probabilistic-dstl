"""Entry point for the pdSTL example cases.

    python src/main.py --all                # the five cases, non-interactive
    python src/main.py --case corridor      # one case
    python src/main.py --case mpc           # MPC consistency check
    python src/main.py --list               # show the available cases

Plots are written to outputs/. Nothing here starts the lane-change or other
planning scenarios; those live in planning/runners.py and are invoked
explicitly.
"""

import argparse
import sys

import matplotlib

from planning.examples import CASES, OUTPUT_DIR, run_all, run_case, run_mpc_check


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="run the five cases")
    group.add_argument(
        "--case",
        metavar="NAME",
        help=f"run one of: {', '.join(sorted(CASES))}, mpc",
    )
    group.add_argument("--list", action="store_true", help="list available cases")
    parser.add_argument("--show", action="store_true", help="display plots")
    parser.add_argument("--no-save", action="store_true", help="do not write plots")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list:
        for name in sorted(CASES):
            print(name)
        print("mpc")
        return 0

    if not args.show:
        matplotlib.use("Agg")  # non-interactive runs need no display

    save = not args.no_save

    if args.all:
        run_all(show=args.show, save=save)
    elif args.case == "mpc":
        run_mpc_check()
    elif args.case in CASES:
        run_case(args.case, show=args.show, save=save)
    else:
        print(f"unknown case {args.case!r}; try --list", file=sys.stderr)
        return 2

    if save:
        print(f"\nPlots written to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
