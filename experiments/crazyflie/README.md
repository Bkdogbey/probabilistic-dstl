# Crazyflie experiment

The deterministic condition replays the supplied `plans/reference.json`:
90 world-frame velocity pairs, held for 0.2 seconds each and refreshed at
20 Hz. The mission lasts 18 seconds. Fitting, planning and analysis run offline.

During the mission, horizontal velocity commands are rotated using measured yaw
and sent through cflib. The onboard controller tracks velocity and height;
there is no horizontal position-error correction toward the reference path.

## Files

| File | Purpose |
| --- | --- |
| `run.py` | Command-line entry point: import, fly, fit, plan and analyze |
| `flight.py` | One ROS Sugar mission using irobot and cflib; preflight, replay and landing |
| `logger.py` | Trial records, onboard timestamps and time alignment |
| `trajectory.py` | Read, validate and save frozen velocity commands |
| `config.py`, `config.example.yml` | Defaults and the setup template; local measurements go in `config.yml` |
| `tracking_model.py` | Fit response lag, gain, drift and covariance; validate on held-out flights |
| `planner.py` | Offline mean-model and pdSTL planning |
| `geometry.py` | Workspace, obstacles, goal and airframe-clearance checks |
| `analysis.py` | Outcome summaries, plan previews and selected flight plots |
| `simulator.py`, `tests/` | Hardware-free replay and regression checks |

## Run a flight

Place the drone within 15 cm of the saved start and leave space to land at the
endpoint. The flight takes off, reaches the start, settles, replays the saved
commands and lands in place.

```bash
conda activate ros_env
source ~/sugar_ws/install/setup.bash
cd ~/Research/probabilistic-dstl/experiments/crazyflie

python run.py fly --fan 0
```

Set the physical fan before running `fly`: **0** is off; **6, 12 and 18** are the
active settings. `--fan` labels the record and selects the model condition; it
does not operate the fan. Connecting does not arm the drone; arming follows
preflight. Ctrl-C during flight requests landing and cleanup. After landing,
press Ctrl-C to exit the idle ROS Sugar component.

Inspect a recorded flight or rehearse without hardware:

```bash
python run.py analyze --trial latest
python run.py fly --fan 0 --dry-run
```

The configured reference is already registered. Use `--plan PATH` to select
another saved plan. Use `import PATH` only to register a new supplied trajectory
after its offline geometry and command-limit checks. The supplied reference
is never optimized or replaced during flight.

## Experiment pipeline

1. Enter measured geometry and airframe radius in `config.yml`, then set
   `measured: true`. For a new checkout, copy `config.example.yml` to `config.yml`.
2. Collect identification flights at fan settings 0, 6, 12 and 18, keeping the
   reference, obstacle layout and fan placement fixed:

   ```bash
   python run.py fly --purpose identification --fan 0
   ```

3. Fit the response and uncertainty model:

   ```bash
   python run.py fit
   ```

   The JSON report in `reports/` contains prediction errors, coverage and
   residual checks. Every third eligible flight is held out. Set `model_report`
   in `config.yml` to the fitted report before planning.
4. Generate and inspect the model-based plans offline for each fan:

   ```bash
   python run.py plan --condition mean_model --fan 0 --output plans/mean_fan0.json
   python run.py plan --condition pdstl --fan 0 --output plans/pdstl_fan0.json
   ```

5. Freeze the reference, fitted model and model-based plans, then collect
   evaluation flights with a fixed run count and balanced condition order:

   ```bash
   python run.py fly --purpose evaluation --fan 0
   python run.py fly --plan plans/mean_fan0.json --purpose evaluation --fan 0
   python run.py fly --plan plans/pdstl_fan0.json --purpose evaluation --fan 0
   python run.py analyze
   ```

The default purpose is `pilot`. Pilots, dry runs and unmeasured trials are
excluded from fitting. Evaluation data is not used to refit the model.
Use `--note` to record unusual events.

## Records and outputs

Each attempt saves `state.csv`, `commands.csv` and `metadata.json` in `logs/`.
Mission scoring ends before landing; historical return legs are also excluded.
A command timestamp records acceptance by cflib, not aircraft acknowledgement.
Landing status records completion of landing commands, without touchdown detection.

The flight checks state freshness, link, workspace, altitude and command timing.
Each new trajectory step has a 50 ms lateness limit. Repeated refreshes may
recover within one trajectory interval, without shifting the schedule or
extending the mission. Altitude checks use a trailing 0.25-second mean.

Plots are limited to plan previews and a selected flight's XY path and height.
Batch analysis writes summary tables in `results/`; fitting writes numerical
validation results in its JSON report. Empirical task-success rates and their
95% intervals are reported separately from model-based planning scores.
Recorded data and generated outputs stay local and are ignored by Git.

## Installation and checks

One-time installation, if needed, from this directory:
`pip install -e "../..[crazyflie]"` and
`pip install -e ~/Research/irobot --no-deps`. ROS Sugar must also be available
in the sourced ROS environment.

From the repository root:

```bash
python -m pytest experiments/crazyflie/tests -q
```
