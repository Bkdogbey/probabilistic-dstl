from experiments.offline import run_always_example, run_eventually_example
from experiments.planning import run_lane_change, run_mpc, run_single_shot
from utils import load_config, skip_run

DEFAULT_CONFIG = "configs/stl_demos.yaml"
config = load_config(DEFAULT_CONFIG)
examples = config["experiments"]["offline_temporal"]
show = config["show_plots"]

# 1. Offline Always operator
with skip_run("run", "Offline Always operator") as check, check():
    always = examples["always"]
    run_always_example(always["threshold"], tuple(always["interval"]), show=show)

# 2. Offline Eventually operator
with skip_run("run", "Offline Eventually operator") as check, check():
    eventually = examples["eventually"]
    run_eventually_example(
        eventually["threshold"], tuple(eventually["interval"]), show=show
    )

# 3. Single-shot planning
with skip_run("skip", "Single-shot planning") as check, check():
    run_single_shot(config_path="configs/scenarios/single_shot.yaml", show=show)

# 4. MPC planning
with skip_run("skip", "MPC planning") as check, check():
    run_mpc(config_path="configs/scenarios/mpc.yaml", show=show)

# 5. Lane-change planning
with skip_run("skip", "Lane-change planning") as check, check():
    run_lane_change(config_path="configs/scenarios/lane_change.yaml", show=show)

# 6. Aggressive lane-change planning
with skip_run("skip", "Aggressive lane-change planning") as check, check():
    run_lane_change(
        config_path="configs/scenarios/lane_change_aggressive.yaml", show=show
    )
