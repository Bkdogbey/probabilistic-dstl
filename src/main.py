"""Single entry point for the selectable pdSTL examples."""

from experiments.offline import (
    run_always_example,
    run_boolean_example,
    run_eventually_example,
)
from experiments.streaming import (
    run_streaming_always_animation,
    run_streaming_always_example,
)
from experiments.until import run_until_example
from utils import load_config, skip_run

DEFAULT_CONFIG = "configs/examples.yml"


def _interval(settings):
    return tuple(settings["interval"])


def main(config_path=DEFAULT_CONFIG, *, show=None):
    """Run examples selected below. Set each skip_run flag to "run" or "skip"."""
    config = load_config(config_path)
    examples = config["experiments"]
    show = config["show_plots"] if show is None else show

    # 1. Boolean probability bounds
    with skip_run("run", "Boolean probability bounds") as check, check():
        run_boolean_example()

    # 2. Offline temporal operators
    with skip_run("skip", "Offline temporal operators") as check, check():
        offline = examples["offline_temporal"]
        always = offline["always"]
        run_always_example(always["threshold"], _interval(always), show=show)
        eventually = offline["eventually"]
        run_eventually_example(
            eventually["threshold"],
            _interval(eventually),
            show=show,
        )

    # 3. Streaming bounded monitor
    with skip_run("skip", "Streaming bounded monitor") as check, check():
        streaming = examples["streaming"]
        if streaming["animate"]:
            run_streaming_always_animation(
                _interval(streaming),
                frame_interval_ms=streaming["frame_interval_ms"],
                repeat=streaming["repeat"],
                show=show,
            )
        else:
            run_streaming_always_example(_interval(streaming), show=show)

    # 4. Safe Until goal
    with skip_run("skip", "Safe Until goal") as check, check():
        until = examples["until"]
        run_until_example(_interval(until), show=show)


if __name__ == "__main__":
    main()
