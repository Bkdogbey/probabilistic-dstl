# Hardware write-up

- `hardware_experiment.tex`: the experiment section (draft). Lines marked
  `TODO(Bennett)` need the author. It needs `booktabs`, `amsmath`, `graphicx`.
- `make_figures.py`: regenerates `figures/*.pdf` and `results_table.tex`
  from the logs. Run it from `experiments/crazyflie`:
  `python writeup/make_figures.py`.

Each group keeps its first 20 flights that started the curve, in flight
order, the same rule as `python run.py analyze --first 20`.
