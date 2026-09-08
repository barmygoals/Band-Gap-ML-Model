"""Builds every feature matrix the master chain needs, so the GPU chain never pays for
featurisation.

Featurisation is pure CPU work, so running it on the CPU partition first means every later
step cache-hits, and it removes the shared-cache write race that would otherwise make the
parallel phase-3 jobs unsafe.

Run: uv run python scripts/prebuild_features.py [--tiers standard physics physics-alt]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXPT_GAP_PATH, PINNED_DATASET, RAW_DATA_PATH

# Every dataset the master phases featurise, in the order they are first needed.
DATASETS = [
    PINNED_DATASET,                                   # phases 1-5 headline
    "cache/filtered_data/8370f06d04b4.csv",           # MD-HIT t0.5
    "cache/filtered_data/63ce6006c352.csv",           # hull<=0 + t0.5 chained
    "cache/controls/hull0_sizematched_t05_seed42.csv",
    "cache/controls/raw_sizematched_t05_seed42.csv",
    RAW_DATA_PATH,                                    # raw-population MD-HIT arm
]


def main(tiers: list[str], include_matbench: bool, include_expt: bool) -> None:
    from analysis import clean_data, load_data, parse_compositions
    from features.descriptors import build_nn_features

    t_start = time.time()
    for path in DATASETS:
        if not os.path.exists(path):
            print(f"SKIP (absent): {path}")
            continue
        comps = parse_compositions(clean_data(load_data(path)))
        for tier in tiers:
            t0 = time.time()
            X = build_nn_features(comps, path, feature_tier=tier)
            print(f"  {os.path.basename(path):<45} tier={tier:<12} "
                  f"{X.shape[0]}x{X.shape[1]}  ({time.time() - t0:.0f}s)")

    if include_matbench:
        # Union matrix per fold, as the consumer keys on the merged CSV (DECISIONS.md 2026-08-19).
        # Through write_fold_union_csv(), so the key comes from the consumer rather than a copy.
        from pymatgen.core import Composition
        import pandas as pd
        from model.matbench_folds import write_fold_union_csv
        fold_dir = os.path.join("cache", "matbench_folds")
        for i in range(5):
            train_csv = os.path.join(fold_dir, f"matbench_fold{i}_train.csv")
            test_csv = os.path.join(fold_dir, f"matbench_fold{i}_test.csv")
            if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
                print(f"SKIP (absent): matbench fold {i}")
                continue
            t0 = time.time()
            train_df = pd.read_csv(train_csv)
            test_df = pd.read_csv(test_csv)
            union_csv = write_fold_union_csv(i, train_df, test_df)
            # Same order as the consumer, train rows then test rows.
            comps = [Composition(f) for f in train_df["formula_pretty"]]
            comps += [Composition(f) for f in test_df["formula_pretty"]]
            X = build_nn_features(comps, union_csv)
            print(f"  {os.path.basename(union_csv):<45} {X.shape[0]}x{X.shape[1]}  ({time.time() - t0:.0f}s)")

    if include_expt and os.path.exists(EXPT_GAP_PATH):
        from model.residual import prepare_expt_features
        for p in (EXPT_GAP_PATH, "datasets/text_mined_bandgap.csv"):
            if os.path.exists(p):
                t0 = time.time()
                X, _, _ = prepare_expt_features(p, dedupe=True)
                print(f"  {os.path.basename(p):<45} {X.shape[0]}x{X.shape[1]}  ({time.time() - t0:.0f}s)")

    print(f"\nFeature pre-build complete in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    # physics_v2 is out of the default, its ladder rung having been disabled 2026-07-22.
    # literature and fractions are near-free, listed so the prebuild covers every ladder matrix.
    p.add_argument("--tiers", nargs="+",
                   default=["standard", "physics", "physics-alt", "literature", "fractions"])
    p.add_argument("--no-matbench", action="store_true")
    p.add_argument("--no-expt", action="store_true")
    a = p.parse_args()
    main(a.tiers, not a.no_matbench, not a.no_expt)
