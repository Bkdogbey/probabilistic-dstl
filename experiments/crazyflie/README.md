# Crazyflie experiment

The experiment supports a 2D baseline path and a 3D figure-eight path. Both
use the same dimension-aware planner.

## Run

Choose the run at the top of `src/main.py`:

```python
ACTION = 'plan'            # plan | fly | analyze
FAN = 12                   # 2 | 6 | 12 | 16
CONDITION = 'pdstl'        # pdstl | deterministic
SCENARIO = 'baseline'      # baseline | figure8
PLOT = False
ANALYSIS = 'latest'        # latest | all | summary
RUN_NUMBER = None
```

Then run:

```bash
python src/main.py
```

- `plan` optimizes and signs the selected pdSTL plan.
- `fly` runs the selected deterministic or pdSTL path.
- `analyze` plots or summarizes recorded flights.

`components/config.yml` contains geometry, dynamics, hardware, safety, and approved
uncertainty values. Changing a planning input invalidates existing plans.

## Safety

Before arming, flight requires a stable position estimate (settled spread,
enough visible base stations per `min_base_stations`) -- the drone can be
sitting anywhere, no specific start position is required. After takeoff it
climbs to `return_z` (above every configured obstacle), transits to the
mission start, then descends to begin the mission. pdSTL flight additionally
requires a current signed plan with positive `rho_after`.

Only the forward experiment is logged. Takeoff, transit-to-start, return, and
landing are outside the trial. Every exit path stops logging, lands when
airborne, disarms, and disconnects.

## Data

```text
logs/2d/             baseline flight logs
logs/3d/             figure-eight flight logs
waypoints/           signed pdSTL plans
plots/               planning and analysis figures
calibration/reports/ reviewed covariance reports
```

## Offline covariance estimation

After repeated deterministic figure-eight flights:

```bash
python experiments/crazyflie/estimate_covariance.py --fans 2 6 12 16
```

The script reads `logs/3d/` and writes a timestamped report. It never runs from
`src/main.py` and never modifies `config.yml`. Review the report, manually copy
approved values into `uncertainty`, record `source_report`, and regenerate the
affected plans.

## Tests

```bash
python -m pytest experiments/crazyflie/tests -q
```

Hardware acceptance remains a manual deterministic-flight check.
