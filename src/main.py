"""Single entry point for the selectable pdSTL examples."""

from experiments.offline import (
    run_always_example,
    run_boolean_example,
    run_eventually_example,
)
from experiments.streaming import (
    run_sliding_always_example,
    run_streaming_always_example,
    run_streaming_eventually_example,
)
from utils import skip_run

# Choose the examples to run.
RUN_BOOLEAN = "run"
RUN_ALWAYS = "run"
RUN_EVENTUALLY = "run"
RUN_SLIDING_ALWAYS = "run"
RUN_STREAMING_ALWAYS = "run"
RUN_STREAMING_EVENTUALLY = "run"

# Offline Gaussian-belief examples. These retain the user's current intervals.
ALWAYS_THRESHOLD = 50.0
ALWAYS_INTERVAL = (0, 1)
EVENTUALLY_THRESHOLD = 55.0
EVENTUALLY_INTERVAL = (0, 1)

# Separate 11-step sliding and streaming examples.
SLIDING_INTERVAL = (0, 5)


def main(
    *,
    show=True,
    run_boolean=RUN_BOOLEAN,
    run_always=RUN_ALWAYS,
    run_eventually=RUN_EVENTUALLY,
    run_sliding_always=RUN_SLIDING_ALWAYS,
    run_streaming_always=RUN_STREAMING_ALWAYS,
    run_streaming_eventually=RUN_STREAMING_EVENTUALLY,
):
    """Run the selected examples; ``show=False`` keeps tests noninteractive."""
    with skip_run(run_boolean, "Boolean operators") as check, check():
        run_boolean_example()

    with skip_run(run_always, "Offline Always") as check, check():
        run_always_example(ALWAYS_THRESHOLD, ALWAYS_INTERVAL, show=show)

    with skip_run(run_eventually, "Offline Eventually") as check, check():
        run_eventually_example(EVENTUALLY_THRESHOLD, EVENTUALLY_INTERVAL, show=show)

    with skip_run(run_sliding_always, "Offline sliding Always") as check, check():
        run_sliding_always_example(SLIDING_INTERVAL, show=show)

    with skip_run(run_streaming_always, "Streaming Always") as check, check():
        run_streaming_always_example(SLIDING_INTERVAL, show=show)

    with skip_run(run_streaming_eventually, "Streaming Eventually") as check, check():
        run_streaming_eventually_example(SLIDING_INTERVAL, show=show)


if __name__ == "__main__":
    main()
