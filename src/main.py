"""Single entry point for the selectable pdSTL examples."""

from experiments.mission import run_mission_example
from experiments.offline import (
    run_always_example,
    run_boolean_example,
    run_eventually_example,
)
from experiments.streaming import (
    run_sliding_always_example,
    run_streaming_always_animation,
    run_streaming_always_example,
    run_streaming_eventually_example,
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


# Boolean example
    with skip_run("run", "Boolean operators") as check, check():
        run_boolean_example()

# Always Example
    always = examples["always"]
    with skip_run("run", "Offline Always") as check, check():
        run_always_example(always["threshold"], _interval(always), show=show)

    eventually = examples["eventually"]
    with skip_run("run", "Offline Eventually") as check, check():
        run_eventually_example(
            eventually["threshold"],
            _interval(eventually),
            show=show,
        )

# Sliding Always Example
    sliding = examples["sliding_always"]
    with skip_run("run", "Offline sliding Always") as check, check():
        run_sliding_always_example(_interval(sliding), show=show)
        

    streaming_always = examples["streaming_always"]
    with skip_run("run", "Streaming Always") as check, check():
        run_streaming_always_example(_interval(streaming_always), show=show)

    streaming_eventually = examples["streaming_eventually"]
    with skip_run("run", "Streaming Eventually") as check, check():
        run_streaming_eventually_example(_interval(streaming_eventually), show=show)

    mission = examples["mission"]
    with skip_run("run", "Composed mission") as check, check():
        run_mission_example(_interval(mission), show=show)

    until = examples["until"]
    with skip_run("run", "Safe until goal") as check, check():
        run_until_example(_interval(until), show=show)

    animation = examples["streaming_animation"]
    with skip_run("skip", "Streaming Always animation") as check, check():
        run_streaming_always_animation(
            _interval(animation),
            frame_interval_ms=animation["frame_interval_ms"],
            repeat=animation["repeat"],
            show=show,
        )


if __name__ == "__main__":
    main()
