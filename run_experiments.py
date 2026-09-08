"""Runs an ordered sequence of experiments and benchmarks from a YAML plan file, which
defaults to experiments/experiment_plan.yaml.

Caching is per step type. The model and data functions are content-addressed, so re-running
after an interruption or a small edit recomputes only what changed. The eight _script_step
types are uncached, as they write fixed paths and never check for a hit (gw_anchor overwrites
its CSV in place).

Each run writes a JSON log under cache/run_logs/. Use --show-log for the most recent, or
--show-log PATH for a specific one.
"""
import argparse
import datetime
import json
import os
import time

import yaml

from cache import CACHE_DIR
from config import DIAGRAMS_FOLDER, PINNED_DATASET, RAW_DATA_PATH
from data.retrieval import retrieve_data
from data.preprocessing import preprocess, default_filtered_data_path
from analysis import run_analysis
from model.training import train
from model.experiments import (
    architecture_search,
    optuna_search_multi_dataset,
    sweep,
    threshold_sweep,
)
from model.benchmark import run_mdhit_comparison, run_matbench_mp_gap, run_benchmark
from model.mdhit_arms import run_mdhit_arms
from model.crabnet_baseline import run_crabnet_baseline
from model.crabnet_data_efficiency import run_crabnet_data_efficiency
from model.crabnet_ood import run_crabnet_ood
from model.crabnet_transfer import run_crabnet_transfer
from model.export_splits import export_splits
from model.extrapolation import run_gap_extrapolation
from model.matbench_folds import run_matbench_from_folds
from model.metal_classifier import run_metal_classifier
from model.residual import train_residual
from model.shap_analysis import run_shap_analysis

def _script_step(module_name: str):
    """Expose a scripts/*.py main() as a plan step type, imported at call time so plain training plans do not pay the xgboost and shap import cost."""
    def _run(**params):
        import importlib
        return importlib.import_module(module_name).main(**params)
    _run.__name__ = module_name
    return _run


REGISTRY = {
    "retrieve_data":     retrieve_data,
    "preprocess":        preprocess,
    "analysis":          run_analysis,
    "train":             train,
    "sweep":             sweep,
    "threshold_sweep":   threshold_sweep,
    "optuna_search_multi_dataset": optuna_search_multi_dataset,
    "activation_width_search": architecture_search,
    "mdhit_comparison":  run_mdhit_comparison,
    # mdhit_arms supersedes mdhit_comparison, scoring three arms on one locked test set.
    # mdhit_comparison stays registered, as results/master_v2_2026-07-30 cites it.
    "mdhit_arms":        run_mdhit_arms,
    "matbench_mp_gap":   run_matbench_mp_gap,
    "ood_benchmark":     run_benchmark,
    "shap_analysis":     run_shap_analysis,
    "train_residual":    train_residual,
    "export_splits":     export_splits,
    "crabnet_baseline":  run_crabnet_baseline,
    "matbench_from_folds":     run_matbench_from_folds,
    "metal_classifier":        run_metal_classifier,
    "gap_extrapolation":       run_gap_extrapolation,
    "crabnet_data_efficiency": run_crabnet_data_efficiency,
    "crabnet_transfer":        run_crabnet_transfer,
    "crabnet_ood":             run_crabnet_ood,
    # dataset gate and analysis-script layer (uncached, see _script_step)
    "verify_dataset":      _script_step("scripts.verify_dataset"),
    # verify_config gates the settings, as an md5 gate cannot tell two cutoffs apart here.
    "verify_config":       _script_step("scripts.verify_config"),
    "xgb_baseline":        _script_step("scripts.xgb_baseline"),
    "xgb_ood":             _script_step("scripts.xgb_ood"),
    "expt_gap_folds":      _script_step("scripts.expt_gap_folds_eval"),
    "gw_anchor":           _script_step("scripts.gw_anchor"),
    "shap_residual":       _script_step("scripts.shap_residual"),
    "false_metal_recovery": _script_step("scripts.false_metal_recovery"),
}

# preprocess and run_analysis have no built-in data path, so these fill one in.
STEP_DEFAULTS = {
    "preprocess": {"input_path": RAW_DATA_PATH},
    "analysis":   {"data_path": PINNED_DATASET},
}

RUN_LOG_DIR = os.path.join(CACHE_DIR, "run_logs")
# CrabNet writes outside cache/, which touched_files would otherwise omit.
_WATCH_DIRS = (CACHE_DIR, DIAGRAMS_FOLDER, os.path.dirname(RAW_DATA_PATH) or ".",
               os.path.join("models", "trained_models"), os.path.join("figures", "lc_data"))


def _snapshot_watch_dirs() -> dict[str, tuple[float, int]]:
    """The (mtime, size) of every watched file, diffed before and after a step to find what it wrote."""
    snapshot: dict[str, tuple[float, int]] = {}
    for root_dir in _WATCH_DIRS:
        if not os.path.isdir(root_dir) or os.path.abspath(root_dir) == os.path.abspath(RUN_LOG_DIR):
            continue
        for dirpath, _, filenames in os.walk(root_dir):
            if os.path.abspath(dirpath).startswith(os.path.abspath(RUN_LOG_DIR)):
                continue  # writing the log itself is not a step's output
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                    snapshot[p] = (st.st_mtime, st.st_size)
                except OSError:
                    pass
    return snapshot


def _summarize_result(result) -> dict:
    """Summarise a step's return value, duck-typed to avoid importing each library, and {} for anything unrecognised."""
    if result is None:
        return {}
    if isinstance(result, str):
        return {"output_path": result}
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        # (model, history) from train()
        history = result[1]
        return {k: history[k] for k in ("test_mae", "test_rmse", "test_r2", "test_mape", "n_train") if k in history}
    if isinstance(result, list) and result and isinstance(result[0], dict):
        # sweep() and threshold_sweep(), a list of per-config or per-threshold dicts
        summary: dict = {"n_results": len(result)}
        if "val_mae" in result[0]:
            summary["best"] = min(result, key=lambda r: r["val_mae"])
        return summary
    if hasattr(result, "trials") and hasattr(result, "directions"):
        # optuna.Study, where best_value raises for a multi-objective study.
        if len(result.directions) > 1:
            pareto = result.best_trials
            # architecture_search() records how the chosen config's tolerance was decided.
            return {
                "n_trials": len(result.trials),
                "n_pareto_trials": len(pareto),
                "pareto_best_val_mae": min((t.values[0] for t in pareto), default=None),
                "pareto_selection": result.user_attrs.get("pareto_selection"),
                "pareto_tolerance_abs": result.user_attrs.get("pareto_tolerance_abs"),
                "pareto_tolerance_source": result.user_attrs.get("pareto_tolerance_source"),
                "chosen_val_mae": result.user_attrs.get("pareto_chosen_val_mae"),
                "chosen_n_params": result.user_attrs.get("pareto_chosen_n_params"),
            }
        return {
            "n_trials": len(result.trials),
            "best_value": result.best_value,
            "best_params": result.best_params,
        }
    if hasattr(result, "columns") and hasattr(result, "__len__"):
        # pandas.DataFrame
        return {"n_rows": len(result), "columns": list(result.columns)}
    return {}


def run_plan(plan_path: str, dry_run: bool = False) -> str | None:
    """Run plan_path top to bottom, returning the written run log's path, or None for a dry run."""
    with open(plan_path) as f:
        steps = yaml.safe_load(f) or []

    enabled_steps = [s for s in steps if s.get("enabled", True)]
    print(f"Loaded {len(steps)} step(s) from {plan_path} ({len(enabled_steps)} enabled)\n")

    log_steps: list[dict] = []
    log_path: str | None = None
    try:
        for i, step in enumerate(steps, 1):
            name, step_type = step["name"], step["type"]
            if step_type not in REGISTRY:
                raise ValueError(f"Unknown step type {step_type!r} in step {i} ({name!r}). "
                                  f"Valid types: {sorted(REGISTRY)}")

            if not step.get("enabled", True):
                print(f"[{i}/{len(steps)}] SKIP  {name}")
                log_steps.append({"name": name, "type": step_type, "enabled": False})
                continue

            params = {**STEP_DEFAULTS.get(step_type, {}), **(step.get("params") or {})}
            print(f"[{i}/{len(steps)}] {'WOULD RUN' if dry_run else 'RUN  '}  {name}  ({step_type})  params={params}")
            if dry_run:
                continue

            before = _snapshot_watch_dirs()
            t0 = time.time()
            result = REGISTRY[step_type](**params)
            duration = time.time() - t0
            after = _snapshot_watch_dirs()
            touched = sorted(
                (
                    {"path": p, "size_kb": round(after[p][1] / 1024, 1)}
                    for p in after if before.get(p) != after[p]
                ),
                key=lambda d: d["path"],
            )

            log_steps.append({
                "name": name, "type": step_type, "enabled": True, "params": params,
                "duration_seconds": round(duration, 1),
                "summary": _summarize_result(result),
                "touched_files": touched,
            })
            print(f"[{i}/{len(steps)}] DONE  {name}  ({duration:.1f}s)\n")
    finally:
        if not dry_run and log_steps:
            os.makedirs(RUN_LOG_DIR, exist_ok=True)
            # Second-resolution names collide once a phase is fanned out as an array job.
            # The suffix is the SLURM array or job id where there is one, and the pid otherwise.
            _tag = (os.environ.get("SLURM_ARRAY_TASK_ID")
                    or os.environ.get("SLURM_JOB_ID") or str(os.getpid()))
            log_path = os.path.join(
                RUN_LOG_DIR,
                datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + f"_{_tag}.json"
            )
            with open(log_path, "w") as f:
                json.dump({"plan_path": plan_path, "steps": log_steps}, f, indent=2, default=str)
            print(f"Run log written to {log_path}")

    return log_path


def print_log(log_path: str | None = None) -> None:
    """Print a run log written by run_plan(), defaulting to the most recent under cache/run_logs/."""
    if log_path is None:
        if not os.path.isdir(RUN_LOG_DIR) or not os.listdir(RUN_LOG_DIR):
            print(f"No run logs found under {RUN_LOG_DIR} -- run the plan first.")
            return
        log_path = os.path.join(RUN_LOG_DIR, sorted(os.listdir(RUN_LOG_DIR))[-1])

    with open(log_path) as f:
        log = json.load(f)

    print(f"Run log: {log_path}")
    print(f"Plan:    {log['plan_path']}\n")

    for i, step in enumerate(log["steps"], 1):
        if not step.get("enabled", True):
            print(f"[{i}] SKIP  {step['name']}")
            continue

        print(f"[{i}] {step['name']}  ({step['type']}, {step.get('duration_seconds', '?')}s)")
        if step.get("params"):
            print(f"     params:  {step['params']}")
        if step.get("summary"):
            print(f"     summary: {step['summary']}")
        touched = step.get("touched_files", [])
        if touched:
            print(f"     touched {len(touched)} file(s):")
            for entry in touched[:5]:
                print(f"       {entry['path']}  ({entry['size_kb']} KB)")
            if len(touched) > 5:
                print(f"       ... and {len(touched) - 5} more")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an ordered experiment plan from a YAML file")
    parser.add_argument("plan", nargs="?", default="experiments/experiment_plan.yaml",
                         help="Path to the YAML plan file (default: experiments/experiment_plan.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the resolved plan without executing anything")
    parser.add_argument("--show-log", nargs="?", const="__latest__", default=None, metavar="PATH",
                         help="Print a run log instead of running the plan. Omit PATH for the "
                              "most recent run (cache/run_logs/), or give a specific log file.")
    args = parser.parse_args()

    if args.show_log is not None:
        print_log(None if args.show_log == "__latest__" else args.show_log)
    else:
        run_plan(args.plan, dry_run=args.dry_run)
