"""Run each experiment by setting its block to "run" or "skip".

Figures are shown, then saved to outputs/ when closed.
"""

from experiments import inspection, lane, reach_avoid
from utils import skip_run

REACH_AVOID = "configs/scenarios/reach_avoid"
LANE = "configs/scenarios/lane"
LIVE = {"live": True, "optimization_every": 1}  # plot and print each iterate


# 1. Reach-avoid: one obstacle
with skip_run("skip", "ReachAvoid") as check, check():
    reach_avoid.run(f"{REACH_AVOID}/obstacle.yaml", **LIVE)


# 2. Reach-avoid (stlpy NarrowPassage): goal A or B past four obstacles
with skip_run("skip", "NarrowPassage") as check, check():
    reach_avoid.run(f"{REACH_AVOID}/narrow_passage.yaml", **LIVE)


# 3. Reach-avoid (stlpy EitherOr): dwell in t1 or t2, then reach the goal
with skip_run("skip", "EitherOr") as check, check():
    reach_avoid.run(f"{REACH_AVOID}/either_or.yaml", **LIVE)


# 4. Lane change into the faster lane
with skip_run("skip", "LaneChange") as check, check():
    lane.run(f"{LANE}/lane_change.yaml", live_optimization=True)


# 5. On-ramp merge into the main lane
with skip_run("skip", "LaneMerge") as check, check():
    lane.run(f"{LANE}/lane_merge.yaml", live_optimization=True)


# 6. Gap or detour: STL on the mean, tightened STL, and pdSTL plan the
#    narrow gap or the long detour at one noise level
with skip_run("skip", "Inspection") as check, check():
    inspection.run(f"{REACH_AVOID}/inspection.yaml", **LIVE)


# 7. Gap or detour across noise levels, in parallel (paper figures)
with skip_run("run", "InspectionSweep") as check, check():
    inspection.sweep(f"{REACH_AVOID}/inspection.yaml")
