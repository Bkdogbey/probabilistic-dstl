

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

config = load_config(DEFAULT_CONFIG)
examples = config["experiments"]
show = config["show_plots"]

# 1. Boolean probability bounds
with skip_run("run", "Boolean probability bounds") as check, check():
    run_boolean_example()

# 2. Offline Always operator
with skip_run("run", "Offline Always operator") as check, check():
    always = examples["offline_temporal"]["always"]
    run_always_example(
        always["threshold"],
        tuple(always["interval"]),
        show=show,
    )

# 3. Offline Eventually operator
with skip_run("run", "Offline Eventually operator") as check, check():
    eventually = examples["offline_temporal"]["eventually"]
    run_eventually_example(
        eventually["threshold"],
        tuple(eventually["interval"]),
        show=show,
    )

# 4. Streaming bounded monitor
with skip_run("skip", "Streaming bounded monitor") as check, check():
    streaming = examples["streaming"]
    interval = tuple(streaming["interval"])

    if streaming["animate"]:
        run_streaming_always_animation(
            interval,
            frame_interval_ms=streaming["frame_interval_ms"],
            repeat=streaming["repeat"],
            show=show,
        )
    else:
        run_streaming_always_example(interval, show=show)

# 5. Safe Until goal
with skip_run("run", "Safe Until goal") as check, check():
    until = examples["until"]
    run_until_example(tuple(until["interval"]), show=show)
