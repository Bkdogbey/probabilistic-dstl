from offline import run_offline_example
from utils import load_config, skip_run


examples = load_config("configs/examples.yaml")
show_plots = examples["show_plots"]


# 1. Always
with skip_run("run", "Always") as check, check():
    run_offline_example("Always", examples["always"], show=show_plots)


# 2. Eventually
with skip_run("run", "Eventually") as check, check():
    run_offline_example(
        "Eventually", examples["eventually"], show=show_plots
    )


# 3. Nested Eventually(Always(predicate))
with skip_run("run", "Nested") as check, check():
    run_offline_example("Nested", examples["nested"], show=show_plots)
