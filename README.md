# Probabilistic differentiable STL

pdSTL evaluates Signal Temporal Logic specifications over uncertain state
trajectories. A model supplies a belief at each prediction step, the belief
evaluates atomic predicates as lower and upper probability bounds, and pdSTL
combines those bounds with Boolean and temporal operators.

```text
belief trajectory → predicate probability intervals → temporal formula trace
```

The included offline examples use Gaussian beliefs to demonstrate `Always` and
`Eventually`; the core `Belief` and `BeliefTrajectory` interfaces are not tied to
Gaussian models, so future estimators may provide their own predicate intervals.

## Install and run

```bash
pip install -r requirements.txt
python src/main.py
```

Examples are selected with the numbered `skip_run` blocks in `src/main.py`.
Their model, predicate, confidence, and temporal-window settings live in
`configs/examples.yaml`. Set `show_plots: false` there for noninteractive runs.

## Package organization

```text
src/
  pdstl/          Generic belief interface and STL operators
  models/         Dynamics and distribution-specific beliefs
  planning/       Deferred optimisation and planning components
  visualization/  Reusable plots
  main.py         Offline operator examples
configs/          Example and planning configuration
```

## License

MIT. See LICENSE.
