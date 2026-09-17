from planning.examples import run_altitude_safety
from planning.runners import run_reach_avoid
from utils import load_config, skip_run

show_plots = load_config("configs/examples.yaml")["show_plots"]


# 1. AltitudeSafety: 1-D altitude, Always[1,H](z >= 50 m)
with skip_run("run", "AltitudeSafety") as check, check():
    print("\nAltitudeSafety")
    run_altitude_safety(show=show_plots, save=True)


# 2. ReachAvoid: 2-D position, Always[1,H](outside obstacle) and Eventually[1,H](inside goal)
with skip_run("run", "ReachAvoid") as check, check():
    print("\nReachAvoid")
    run_reach_avoid(show=show_plots, save=True)
