"""Phase-0 gate for the settings that decide which rows a campaign trains on.

The dataset gates prove a file is byte-identical to what the reported numbers used, not that
it is the file this campaign's config would build. On the hull<=0 corpus, MIN_ELEMENT_COUNT 6
and 20 give byte-identical files, so a campaign could run at 20 whilst every arm trained on a
corpus built at 6 and no md5 gate would notice.

It also checks the pinned dataset agrees across config.PINNED_DATASET, each phase plan's
data_path and gate 1's own, all repointed by hand on 2026-08-17.

Run standalone:  uv run python scripts/verify_config.py --plans-dir experiments/master_v2
As a plan step:  type: verify_config
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

import config
from cache import file_md5, get_cache_path
from data.preprocessing import _PREPROCESSING_PY
from scripts.build_element_filtered import CORPORA

_ELEMENT_FILTERED_DIR = os.path.join("cache", "element_filtered")


def _expected_path(source: str, min_element_count: int) -> str:
    """Where filter_rare_elements would write this corpus at this cutoff, mirroring
    scripts/build_element_filtered.py's call so a new keyword there forces this gate to move
    with it."""
    settings = {"min_element_count": min_element_count, "formula_column": "formula_pretty"}
    return get_cache_path("element_filtered", [source], settings,
                          extra_files=(_PREPROCESSING_PY,))


def _norm(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")


def _plan_data_paths(plans_dir: str) -> list[tuple[str, str, str]]:
    """(plan file, step name, data_path) for every enabled step naming a data_path under
    cache/element_filtered, disabled steps being skipped on purpose so a retired plan may keep
    an old path (the phase3c/3d rule)."""
    found = []
    for fname in sorted(os.listdir(plans_dir)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(plans_dir, fname)
        with open(path, encoding="utf-8") as f:
            plan = yaml.safe_load(f) or []
        steps = plan if isinstance(plan, list) else plan.get("steps", [])
        for step in steps:
            if not isinstance(step, dict) or not step.get("enabled"):
                continue
            dp = (step.get("params") or {}).get("data_path")
            if isinstance(dp, str) and "element_filtered" in dp:
                found.append((fname, step.get("name", "?"), dp))
    return found


def verify_config(
    plans_dir: str = "experiments/master_v2",
    expect_min_element_count: int | None = None,
    gate_plan: str = "phase0_verify.yaml",
) -> dict:
    """Assert the campaign's row-selection settings agree with the artifacts on disk, where
    expect_min_element_count is pinned in the plan rather than read from config, else a gate
    reading its expectation from the thing it checks would pass whatever config said."""
    failures: list[str] = []
    mec = config.MIN_ELEMENT_COUNT
    print(f"config.MIN_ELEMENT_COUNT = {mec}")

    if expect_min_element_count is not None and mec != expect_min_element_count:
        failures.append(
            f"MIN_ELEMENT_COUNT drift: config says {mec}, this plan expects "
            f"{expect_min_element_count}. Either the constant changed without the plan, "
            f"or this plan belongs to an earlier campaign."
        )

    # 1. every filtered corpus exists at the cutoff config names
    built: dict[str, str] = {}
    for label, source in CORPORA:
        expected = _expected_path(source, mec)
        if not os.path.exists(expected):
            failures.append(
                f"{label}: no corpus built at MIN_ELEMENT_COUNT={mec} (expected "
                f"{_norm(expected)}). Run: uv run python scripts/build_element_filtered.py"
            )
            continue
        built[label] = expected
        print(f"  {label:20s} {_norm(expected)}  md5 {file_md5(expected)[:8]}…")

    # 2. config.PINNED_DATASET is the pinned corpus at this cutoff, not one of equal content.
    pinned_label, pinned_source = CORPORA[0]
    expected_pinned = built.get(pinned_label)
    if expected_pinned and _norm(config.PINNED_DATASET) != _norm(expected_pinned):
        same = (os.path.exists(config.PINNED_DATASET)
                and file_md5(config.PINNED_DATASET) == file_md5(expected_pinned))
        failures.append(
            f"PINNED_DATASET drift: config points at {_norm(config.PINNED_DATASET)}, "
            f"MIN_ELEMENT_COUNT={mec} implies {_norm(expected_pinned)}"
            + (" — SAME CONTENT, so no dataset gate would catch this; the campaign would "
               "run at a cutoff its config does not state." if same else "")
        )

    # 3. every enabled plan step, and gate 1, name the file config names
    pinned = _norm(config.PINNED_DATASET)
    known = {_norm(p) for p in built.values()}
    for fname, step_name, dp in _plan_data_paths(plans_dir):
        if _norm(dp) not in known:
            failures.append(
                f"{fname} :: {step_name}: data_path {_norm(dp)} is not a corpus built at "
                f"MIN_ELEMENT_COUNT={mec} ({', '.join(sorted(known))})"
            )
    gate_file = os.path.join(plans_dir, gate_plan)
    if os.path.exists(gate_file):
        with open(gate_file, encoding="utf-8") as f:
            gate_steps = yaml.safe_load(f) or []
        gate1 = next((s for s in gate_steps
                      if isinstance(s, dict) and s.get("type") == "verify_dataset"
                      and s.get("enabled")
                      and "element_filtered" in ((s.get("params") or {}).get("data_path") or "")), None)
        if gate1 is None:
            failures.append(f"{gate_plan}: no enabled element_filtered dataset gate found")
        elif _norm(gate1["params"]["data_path"]) != pinned:
            failures.append(
                f"{gate_plan} :: {gate1['name']} gates {_norm(gate1['params']['data_path'])} "
                f"but config.PINNED_DATASET is {pinned} — the gate would verify a file the "
                f"campaign does not train on"
            )

    n_steps = len(_plan_data_paths(plans_dir))
    if failures:
        raise RuntimeError(
            "Config/plan agreement FAILED:\n  - " + "\n  - ".join(failures))
    print(f"  pinned dataset agrees across config, {n_steps} enabled plan step(s) and the "
          f"phase-0 gate")
    return {"min_element_count": mec, "corpora": {k: _norm(v) for k, v in built.items()},
            "plan_steps_checked": n_steps}


def main(**params) -> dict:  # plan-step entry point (run_experiments._script_step)
    return verify_config(**params)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plans-dir", default="experiments/master_v2")
    p.add_argument("--expect-min-element-count", type=int, default=None)
    a = p.parse_args()
    verify_config(plans_dir=a.plans_dir, expect_min_element_count=a.expect_min_element_count)
