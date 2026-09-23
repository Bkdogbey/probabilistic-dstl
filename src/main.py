"""Run the current projects by changing each block's ``run`` or ``skip`` flag."""

from planning.runners import (
    run_altitude_safety,
    run_lane_change,
    run_reach_avoid,
)
from utils import skip_run


# A displayed figure or animation is shown first; closing it starts saving.
show_plots = True
save_plots = True
live_plots = True

# Stream sampled gradient-descent steps and open their live convergence plot.
show_optimization = True
optimization_every = 5  # set to 1 to display every gradient update


# 1. Altitude safety: Always[1,H](altitude >= threshold)
with skip_run("skip", "AltitudeSafety") as check, check():
    run_altitude_safety(
        show=show_plots,
        save=save_plots,
        live_optimization=show_optimization,
        optimization_every=optimization_every,
    )


# 2. One-shot reach-avoid planning
with skip_run("run", "ReachAvoid") as check, check():
    run_reach_avoid(
        show=show_plots,
        save=save_plots,
        live=live_plots,
        optimization_every=optimization_every,
    )


# 3. Two-lane change
with skip_run("skip", "LaneChange") as check, check():
    run_lane_change(
        show=show_plots,
        save=save_plots,
        live=live_plots,
        live_optimization=show_optimization,
        optimization_every=optimization_every,
    )


# 4. On-ramp merge into the main lane
with skip_run("skip", "LaneMerge") as check, check():
    run_lane_change(
        "configs/scenarios/lane_merge.yaml",
        show=show_plots,
        save=save_plots,
        live=live_plots,
        live_optimization=show_optimization,
        optimization_every=optimization_every,
    )
