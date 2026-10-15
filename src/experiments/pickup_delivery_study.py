"""Blocks, records, and resume for the pickup-delivery study.

    python -m experiments.pickup_delivery_study <block> [--exp ID] [--workers N]
    python -m experiments.pickup_delivery_study plots --exp ID [--exp ID ...]
    python -m experiments.pickup_delivery_study estimate --exp pilot

Blocks are declared in the scenario YAML. Each trial appends one JSON line to
outputs/pickup_delivery/<exp>/records.jsonl (artifacts in artifacts/). A rerun
skips finished keys and refuses an experiment whose config hash changed.
Summaries are always recomputed from the rows, never appended.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.stats import kendalltau

from experiments import pickup_delivery as pdl
from models.rollouts import create_gaussian_belief_trajectory
from utils import load_config

CONFIG = "configs/scenarios/pickup_delivery/warehouse.yaml"
ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs" / "pickup_delivery"
BASE_BRANCH = "origin/RA_L-planning"


# --- Bookkeeping -----------------------------------------------------------------


def _git(*args):
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance():
    """Base and current commits, dirty flag, and the compute environment."""
    return {
        "base_sha": _git("merge-base", "HEAD", BASE_BRANCH),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--", "src", "configs")),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_threads_per_worker": 1,
        "dtype": "float32",
    }


def config_hash(cfg, block):
    text = json.dumps({"cfg": cfg, "block": block}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def read_records(directory):
    path = Path(directory) / "records.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def open_experiment(exp_id, cfg, block_name, block):
    """Create or reopen an experiment directory; refuse a changed config."""
    directory = OUTPUTS / exp_id
    manifest_path = directory / "manifest.json"
    digest = config_hash(cfg, block)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config_hash"] != digest:
            raise SystemExit(
                f"{exp_id}: config hash {digest} differs from the recorded "
                f"{manifest['config_hash']}; use a new --exp id"
            )
        return directory, manifest
    (directory / "artifacts").mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": exp_id,
        "block": block_name,
        "block_config": block,
        "config_hash": digest,
        "config": cfg,
        **provenance(),
    }
    return directory, manifest


def write_manifest(directory, manifest):
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=1, default=str), encoding="utf-8"
    )


def seed(cfg, base, *parts):
    """Deterministic seed from a named base and integer parts."""
    entropy = [int(cfg[base]), *[int(p) for p in parts]]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def instance_ids(cfg, spec):
    if isinstance(spec, str):
        first, last = cfg["instances"][f"{spec}_ids"]
        return list(range(first, last + 1))
    return [int(i) for i in spec]


def draw_instances(cfg, ids):
    """Instances plus every rejected draw (with reasons) for the manifest."""
    instances, rejected = {}, []
    for i in ids:
        instances[i], rejects = pdl.draw_instance(cfg, i)
        rejected += rejects
    return instances, rejected


# --- Trial blocks ----------------------------------------------------------------


def _variants(cfg, block):
    """Method configs for the block, expanding a tuning grid into variants."""
    methods = dict(cfg["methods"])
    for base, grid in (block.get("grid") or {}).items():
        for name, values in grid.items():
            for value in values:
                method = json.loads(json.dumps(cfg["methods"][base]))
                if name == "lr":
                    method.setdefault("planner", {})["lr"] = value
                else:
                    method[name] = value
                methods[f"{base}[{name}={value}]"] = method
    return methods


def trial_tasks(cfg, block):
    """Every (instance, noise, spec, method, H) trial a block declares."""
    seed_base = block.get("seed", "evaluation_seed")
    horizons = block.get("horizons", [cfg["H"]])
    noises, specs = list(cfg["noise"]), list(pdl.SPECS)
    grid_methods = [m for m in cfg["methods"] if "[" in m]
    tasks = []
    for run in block["runs"]:
        methods = run.get("methods", grid_methods)
        for i, noise, spec, method, horizon in product(
            instance_ids(cfg, run["instances"]),
            run["noise"],
            run["specs"],
            methods,
            horizons,
        ):
            parts = (i, noises.index(noise), specs.index(spec), horizon)
            tasks.append(
                {
                    "key": f"{i}|{noise}|{spec}|{method}|H{horizon}",
                    "instance": i,
                    "noise": noise,
                    "spec": spec,
                    "method": method,
                    "H": horizon,
                    "eval_seed": seed(cfg, seed_base, *parts),
                    "plan_seed": seed(cfg, "plan_seed", *parts) % (2**31 - 1),
                }
            )
    return tasks


def _run_task(cfg, block, instance, task, artifact_dir):
    """Worker: one trial, always returning a record (errors included)."""
    torch.set_num_threads(1)
    started = perf_counter()
    try:
        record, artifact = pdl.run_trial(
            cfg,
            instance,
            task["noise"],
            task["spec"],
            task["method"],
            budget=block["budget_s"],
            rollouts=block["rollouts"],
            eval_seed=task["eval_seed"],
            plan_seed=task["plan_seed"],
            horizon=task["H"],
        )
    except Exception:  # noqa: BLE001 - every attempted trial is kept
        record = {"status": "error", "error": traceback.format_exc()}
        artifact = {}
    if artifact:
        name = task["key"].replace("|", "_").replace("[", "(")
        name = name.replace("]", ")").replace("=", "-") + ".pt"
        torch.save(artifact, Path(artifact_dir) / name)
        record["artifact"] = f"artifacts/{name}"
    record.update(task, wall_s=perf_counter() - started)
    return record


def _diag_task(cfg, block, instance, task, artifact_dir):
    """Worker: score-usefulness candidates for one instance/noise."""
    torch.set_num_threads(1)
    try:
        rows = score_candidates(cfg, block, instance, task)
    except Exception:  # noqa: BLE001 - every attempted trial is kept
        rows = [{"status": "error", "error": traceback.format_exc()}]
    return [{**task, **row} for row in rows]


def run_block(block_name, exp_id, workers, config_path=CONFIG):
    """Run (or resume) the trials of one block and append their records."""
    cfg = load_config(config_path)
    block = cfg["blocks"][block_name]
    cfg = {**cfg, "methods": _variants(cfg, block)}
    directory, manifest = open_experiment(exp_id, cfg, block_name, block)
    tasks = trial_tasks(cfg, block)
    ids = sorted({task["instance"] for task in tasks})
    instances, rejected = draw_instances(cfg, ids)
    manifest.update(instances=instances, rejected_draws=rejected)
    write_manifest(directory, manifest)

    done = {r["key"] for r in read_records(directory) if "key" in r}
    pending = [task for task in tasks if task["key"] not in done]
    worker = _diag_task if block_name == "score_diag" else _run_task
    print(
        f"{exp_id}: {len(tasks)} trials, {len(tasks) - len(pending)} done, "
        f"{len(pending)} to run on {workers} workers",
        flush=True,
    )
    records_path = directory / "records.jsonl"
    jobs = [
        (cfg, block, instances[t["instance"]], t, str(directory / "artifacts"))
        for t in pending
    ]
    for rows in _execute(worker, jobs, workers):
        rows = rows if isinstance(rows, list) else [rows]
        with records_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=float) + "\n")
        _report(rows[0])
    return directory


def _execute(worker, jobs, workers):
    """Yield results as they finish; one worker runs in-process (no spawn)."""
    if workers == 1:
        for job in jobs:
            yield worker(*job)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, *job) for job in jobs]
        for future in as_completed(futures):
            yield future.result()


def _report(record):
    if record.get("status") != "returned":
        print(f"  {record['key']}: {record.get('status')}", flush=True)
        return
    if "candidate" in record:
        print(f"  {record['key']}: {record['candidates']} candidates")
        return
    execution, scores = record["execution"], record["pdstl"]
    print(
        f"  {record['key']}: hard {record['initial_hard_lower']:.3f} -> "
        f"{scores['hard_lower']:.3f}, success {execution['rate']:.3f}, "
        f"{record['total_s']:.0f} s",
        flush=True,
    )


# --- Score usefulness diagnostic -------------------------------------------------


def score_candidates(cfg, block, instance, task):
    """Hard pdSTL score vs empirical full-task success on a candidate pool.

    The pool is the shared start bank plus every `stride`-th pdSTL iterate
    from each start; success uses diagnostic seeds only.
    """
    s = pdl.build(cfg, instance, task["noise"], task["H"])
    spec = pdl.specification(s.env, s.cfg["H"], s.cfg["dwell"], task["spec"])
    method = cfg["methods"]["pdstl"]
    pool = []
    for name, start in pdl.start_bank(s, task["spec"]).items():
        pool.append((name, -1, start.controls))

        def keep(iteration, record, name=name):
            if (iteration + 1) % block["stride"] == 0:
                pool.append((name, iteration, record.controls))

        pdl.plan_gradient(
            s,
            method,
            spec,
            start.controls,
            block["budget_s"] / len(cfg["starts"]),
            on_iteration=keep,
        )
    rows = []
    for k, (name, iteration, controls) in enumerate(pool):
        execution = pdl.execute(
            s, controls, block["rollouts"], task["eval_seed"], task["spec"]
        )
        rows.append(
            {
                "status": "returned",
                "candidate": k,
                "candidates": len(pool),
                "start": name,
                "iteration": iteration,
                "hard_lower": pdl.pdstl_scores(s, spec, controls)[0][
                    "hard_lower"
                ],
                "success_rate": execution["rate"],
                "successes": execution["successes"],
                "samples": execution["samples"],
            }
        )
    return rows


def summarize_scores(records):
    """Within-condition ranking agreement, tie and zero-score rates."""
    groups = {}
    for r in records:
        if r.get("status") == "returned":
            groups.setdefault((r["instance"], r["noise"]), []).append(r)
    summary = []
    for (instance, noise), rows in sorted(groups.items()):
        hard = np.array([r["hard_lower"] for r in rows])
        rate = np.array([r["success_rate"] for r in rows])
        tau = kendalltau(hard, rate).statistic if np.ptp(hard) else np.nan
        tied = np.mean([np.sum(hard == h) > 1 for h in hard])
        summary.append(
            {
                "instance": instance,
                "noise": noise,
                "candidates": len(rows),
                "kendall_tau": float(tau),
                "tie_rate": float(tied),
                "zero_rate": float(np.mean(hard == 0)),
            }
        )
    return summary


# --- Monitor scaling -------------------------------------------------------------


def _family(env, length, dwell, family):
    if family == "GF":
        return pdl.specification(env, length, dwell, "R1")
    if family == "FG":
        from pdstl.operators import Always, Eventually
        from planning.environment import inside

        held = Always(inside(env.region("P1")), [0, dwell])
        return Eventually(held, [0, length - dwell])
    return pdl.specification(env, length, dwell, "R3")


def _peak_bytes(fn):
    """Peak live CPU tensor bytes inside fn, from profiler self allocations."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as p:
        fn()
    events = sorted(
        (e for e in p.events() if e.self_cpu_memory_usage),
        key=lambda e: e.time_range.start,
    )
    level = peak = 0
    for event in events:
        level += event.self_cpu_memory_usage
        peak = max(peak, level)
    return int(peak)


def scaling(cfg, block):
    """Monitor-only forward/backward time and memory vs trace length.

    Belief traces come from one fixed random-control rollout (excluded from
    timing). The output is the origin-0 score; backward is d(smooth)/d(mean).
    """
    torch.set_num_threads(1)
    instance, _ = pdl.draw_instance(cfg, 0)
    rows = []
    for length, family in product(block["lengths"], block["families"]):
        s = pdl.build(cfg, instance, "medium", horizon=length)
        formula = _family(s.env, length, s.cfg["dwell"], family)
        generator = torch.Generator().manual_seed(length)
        controls = 0.3 * torch.randn(length, 2, generator=generator)
        with torch.no_grad():
            mean, cov = s.dyn(controls, *s.state)
        mean = mean.detach()

        def forward(beta):
            leaf = mean.clone().requires_grad_(True)
            beliefs = create_gaussian_belief_trajectory(leaf[0], cov[0])
            return leaf, formula(beliefs, beta=beta)[0, 0, 0]

        def timed(fn):
            for _ in range(block["warmup"]):
                fn()
            times = []
            for _ in range(block["repeats"]):
                started = perf_counter()
                fn()
                times.append(perf_counter() - started)
            return float(np.median(times))

        def smooth_backward():
            leaf, value = forward(block["beta"])
            value.backward()

        smooth_forward = timed(lambda: forward(block["beta"]))
        rows.append(
            {
                "length": length,
                "family": family,
                "lookahead": pdl.lookahead(formula),
                "hard_forward_s": timed(lambda: forward(None)),
                "smooth_forward_s": smooth_forward,
                "backward_s": timed(smooth_backward) - smooth_forward,
                "peak_bytes_forward_backward": _peak_bytes(smooth_backward),
                "output": "origin-0 score",
                "batch": 1,
                "threads": torch.get_num_threads(),
                "dtype": str(mean.dtype),
                "warmup": block["warmup"],
                "repeats": block["repeats"],
                "timing": "median wall time",
            }
        )
        print(
            f"  T={length} {family}: fwd {smooth_forward * 1e3:.0f} ms, "
            f"bwd {rows[-1]['backward_s'] * 1e3:.0f} ms",
            flush=True,
        )
    return rows


def run_scaling(exp_id, config_path=CONFIG):
    cfg = load_config(config_path)
    block = cfg["blocks"]["scaling"]
    directory, manifest = open_experiment(exp_id, cfg, "scaling", block)
    write_manifest(directory, manifest)
    rows = scaling(cfg, block)
    with (directory / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return directory


# --- Cost estimate ---------------------------------------------------------------


def estimate(exp_id, workers, config_path=CONFIG):
    """Projected wall time of the full blocks from measured pilot trials."""
    cfg = load_config(config_path)
    records = [
        r
        for r in read_records(OUTPUTS / exp_id)
        if r.get("status") == "returned" and "execution" in r
    ]
    if not records:
        raise SystemExit(f"{exp_id}: no returned trials to measure")
    overhead = np.mean([r["wall_s"] - r["budget_s"] for r in records])
    per_rollout = np.mean(
        [r["execution_s"] / r["execution"]["samples"] for r in records]
    )
    lines = [
        f"measured on {len(records)} trials of {exp_id}: overhead "
        f"{overhead:.1f} s/trial beyond the budget (starts, probes, scoring), "
        f"{per_rollout * 1e3:.2f} ms per execution rollout"
    ]
    total = 0.0
    for name in ("tune", "main", "spec", "horizon", "score_diag"):
        block = cfg["blocks"][name]
        methods = _variants(cfg, block)
        tasks = trial_tasks({**cfg, "methods": methods}, block)
        seconds = len(tasks) * (
            block["budget_s"] + overhead + block["rollouts"] * per_rollout
        )
        total += seconds
        lines.append(
            f"{name:10s} {len(tasks):4d} trials  {seconds / 3600:6.1f} CPU-h  "
            f"{seconds / 3600 / workers:6.1f} h on {workers} workers"
        )
    lines.append(
        f"total      {total / 3600:6.1f} CPU-h  {total / 3600 / workers:6.1f} h"
    )
    text = "\n".join(lines)
    print(text)
    (OUTPUTS / exp_id / "estimate.txt").write_text(
        text + "\n", encoding="utf-8"
    )
    return text


# --- CLI -------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "block",
        choices=[
            "smoke",
            "tune",
            "pilot",
            "main",
            "spec",
            "horizon",
            "score_diag",
            "scaling",
            "plots",
            "estimate",
        ],
    )
    parser.add_argument("--exp", action="append", help="experiment id(s)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args(argv)
    exp_ids = args.exp or [args.block]
    if args.block == "plots":
        from visualization.pickup_delivery_figures import make_all

        make_all([OUTPUTS / e for e in exp_ids], args.config)
    elif args.block == "estimate":
        estimate(exp_ids[0], args.workers, args.config)
    elif args.block == "scaling":
        run_scaling(exp_ids[0], args.config)
    else:
        run_block(args.block, exp_ids[0], args.workers, args.config)


if __name__ == "__main__":
    main()
