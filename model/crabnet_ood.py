"""CrabNet through the full shared OOD artifact, reading the same fold definitions FlexNet
and XGBoost consume from model/folds.py so the three are row-for-row comparable.

The headline comparison is in-distribution, whilst this tests whether the attention model's
advantage survives distribution shift. It also captures what the baseline runs did not
persist, per-prediction uncertainty from CrabNet's aleatoric sigma and learning curves
harvested from the fitted object.

One seed per fold by default, as each fold is a full CrabNet fit. Folds with too few rows are
skipped with a printed reason.

Run: python -m model.crabnet_ood, or a "crabnet_ood" plan step. GPU recommended.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from cache import get_cache_path, write_meta
from data.preprocessing import default_filtered_data_path

from config import PINNED_DATASET as _DEFAULT_FILTERED_PATH
from model.crabnet_baseline import _cost, _harvest_curves, _load_crabnet, _metrics

_CRABNET_OOD_PY = os.path.abspath(__file__)
_FOLDS_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "folds.py")


def run_crabnet_ood(
    data_path: str = _DEFAULT_FILTERED_PATH,
    threshold_ev: float = 4.0,
    seed: int = 42,
    epochs: int | None = None,
    force_cpu: bool = False,
    min_test: int = 20,
    min_train: int = 200,
    max_folds: int | None = None,
    train_subset: str | None = None,
    only_folds: list[str] | None = None,
    fold_index: int | None = None,
    resume: bool = True,
    aggregate_only: bool = False,
    list_only: bool = False,
) -> pd.DataFrame | None:
    """Every eligible fold of the shared OOD artifact, each a fresh default CrabNet fit.

    Returns per-fold metrics and caches results, predictions with sigma, and harvested curves
    under cache/crabnet_ood/.

    train_subset swaps each fold's training side for the "dedup" or "sizematched" arm, leaving
    test sides untouched, and is in the cache key so the arms coexist.

    only_folds and fold_index select a subset, and each fold's metrics are cached as it
    finishes, so a wall-kill costs the fold in flight rather than the phase and the suite can
    run as an array job. They are kept out of the cache key, as every shard has to resolve to
    the same hash (DECISIONS.md 2026-08-14).
    """
    CrabNet = _load_crabnet()
    if CrabNet is None:
        return None

    settings = {
        "threshold_ev": threshold_ev, "seed": seed, "epochs": epochs,
        "min_test": min_test, "min_train": min_train, "max_folds": max_folds,
    }
    # Only when set, so the full-data arm keeps its existing hash.
    if train_subset:
        settings["train_subset"] = train_subset
    # _FOLDS_PY in extra_files, so a changed split recipe invalidates these results.
    cache_path = get_cache_path(
        "crabnet_ood", [data_path], settings,
        extra_files=(_CRABNET_OOD_PY, _FOLDS_PY), ext="csv"
    )
    out_dir = os.path.dirname(cache_path)
    if os.path.exists(cache_path):
        print(f"Cache valid — loading CrabNet OOD results from {cache_path}")
        return pd.read_csv(cache_path)

    df = pd.read_csv(data_path, usecols=["formula_pretty", "band_gap"]).dropna()
    formulas = df["formula_pretty"].to_numpy()
    y = df["band_gap"].to_numpy(dtype="float64")
    print(f"CrabNet OOD on {data_path}: {len(y)} rows")

    # Folds come from the shared artifact (2026-07-27) rather than being rebuilt here.
    # The old inline control was redrawn per seed, so pre-2026-07-27 numbers mix folds.
    # formulas= asserts row ordering, as this module reads the CSV directly.
    from model.folds import build_ood_folds, load_ood_folds
    from model.train_subsets import apply_subset, resolve_train_subset
    folds_dir = build_ood_folds(data_path, threshold_ev=threshold_ev)
    # Training side only, te stays the artifact's.
    _subs = resolve_train_subset(train_subset, data_path, formulas, threshold_ev=threshold_ev)
    folds = [(f"{split}_{fold}", apply_subset(_subs, split, fold, tr), te)
             for split, fold, tr, te in load_ood_folds(folds_dir, formulas=formulas)]
    if max_folds is not None:
        folds = folds[:max_folds]

    # Eligibility decided before any selection, so every shard expects the same fold set.
    eligible = [(lab, tr, te) for lab, tr, te in folds
                if len(te) >= min_test and len(tr) >= min_train]
    eligible_labels = {lab for lab, _, _ in eligible}
    for lab, tr, te in folds:
        if lab not in eligible_labels:
            print(f"  [{lab}] not eligible (train={len(tr)}, test={len(te)})")
    print(f"{len(folds)} folds from {folds_dir}; {len(eligible)} eligible")

    if list_only:
        for i, (lab, tr, te) in enumerate(eligible):
            print(f"{i}\t{lab}\ttrain={len(tr)}\ttest={len(te)}")
        return None
    if aggregate_only:
        return aggregate_crabnet_ood(cache_path, [lab for lab, _, _ in eligible], seed,
                                     data_path, settings)

    todo = eligible
    if fold_index is not None:
        if not 0 <= fold_index < len(eligible):
            # Exit 0 rather than an error, as an array is sized before the artifact is built.
            # Under-provisioning is caught by the aggregation step instead.
            print(f"fold-index {fold_index} >= {len(eligible)} eligible folds — nothing to do")
            return None
        todo = [eligible[fold_index]]
    elif only_folds is not None:
        want = set(only_folds)
        todo = [f for f in eligible if f[0] in want]
        missing = want - {f[0] for f in eligible}
        if missing:
            raise KeyError(f"unknown or ineligible fold label(s): {sorted(missing)}")
    if todo is not eligible:
        print(f"  running {len(todo)} of them: {', '.join(f[0] for f in todo)}")

    fold_dir = cache_path.replace(".csv", "_folds")
    os.makedirs(fold_dir, exist_ok=True)

    for fold_label, tr_idx, te_idx in todo:
        fold_json = os.path.join(fold_dir, f"{fold_label}_seed{seed}.json")
        if resume and os.path.exists(fold_json):
            print(f"  [{fold_label}] cached — skipping")
            continue
        # CrabNet's checkpoint-selection validation set, 10% of train, capped 2000, floored 32.
        # Carved from train, so test purity matches the FlexNet suite.
        rng_f = np.random.default_rng(seed)
        tr_shuf = rng_f.permutation(tr_idx)
        n_val = max(min(len(tr_shuf) // 10, 2000), 32)
        val_idx, fit_idx = tr_shuf[:n_val], tr_shuf[n_val:]

        def _df(idx):
            return pd.DataFrame({"formula": formulas[idx], "target": y[idx]})

        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Seed and arm both in the name, as these three outputs are not hash-scoped.
        # Without the arm, concurrent arm arrays collide on one .pth per fold and seed.
        # Suffixed only when set, so the full-data arm keeps its old names.
        _tag = f"{fold_label}_seed{seed}" + (f"_{train_subset}" if train_subset else "")
        kwargs: dict = {"mat_prop": "band_gap", "losscurve": False, "learningcurve": False,
                        "force_cpu": force_cpu, "verbose": True, "random_state": seed,
                        "model_name": f"crabnet_ood_{_tag}"}
        if epochs is not None:
            kwargs["epochs"] = epochs
        cb = CrabNet(**kwargs)
        t0 = time.perf_counter()
        cb.fit(_df(fit_idx), _df(val_idx))
        train_seconds = time.perf_counter() - t0

        out = cb.predict(_df(te_idx), return_uncertainty=True)
        pred, sigma = (np.asarray(out[0], dtype="float64"), np.asarray(out[1], dtype="float64")) \
            if isinstance(out, tuple) else (np.asarray(out, dtype="float64"), np.full(len(te_idx), np.nan))

        metrics = _metrics(y[te_idx], pred)
        print(f"  [{fold_label}] train={len(fit_idx)} test={len(te_idx)}  "
              f"MAE={metrics['mae']:.4f} eV  R²={metrics['r2']:.4f}  [{train_seconds:.1f}s]")
        row = {"fold": fold_label, "n_train": len(fit_idx), "n_test": len(te_idx), **metrics,
               **_cost(cb, train_seconds, len(fit_idx))}

        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({
            "formula": formulas[te_idx], "y_true": y[te_idx], "y_pred": pred, "sigma": sigma,
        }).to_csv(os.path.join(out_dir, f"preds_{_tag}.csv"), index=False)
        _harvest_curves(cb, _tag, out_dir)
        # Written last, as this file is the resume gate and must not outlive its own outputs.
        with open(fold_json, "w", encoding="utf-8") as fh:
            json.dump(row, fh)

    return aggregate_crabnet_ood(cache_path, [lab for lab, _, _ in eligible], seed,
                                 data_path, settings)


def aggregate_crabnet_ood(cache_path: str, expected: list[str], seed: int,
                          data_path: str, settings: dict) -> pd.DataFrame | None:
    """Assemble the per-fold JSONs into the results CSV, only once every eligible fold is
    present, as a partial CSV is indistinguishable from a complete one to master_summary.py
    (the 2026-07-28 wall-kill)."""
    fold_dir = cache_path.replace(".csv", "_folds")
    have, missing = [], []
    for lab in expected:
        p = os.path.join(fold_dir, f"{lab}_seed{seed}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                have.append(json.load(fh))
        else:
            missing.append(lab)

    results = pd.DataFrame(have)
    if missing:
        print(f"\n=== CrabNet OOD: {len(have)}/{len(expected)} folds done, "
              f"{len(missing)} outstanding ===")
        print("  missing: " + ", ".join(missing))
        print(f"  aggregate NOT written; rerun the missing folds, then aggregate with:\n"
              f"    uv run python -m model.crabnet_ood --seed {seed} --aggregate-only")
        return results if len(results) else None

    print("\n=== CrabNet OOD summary (all folds present) ===")
    print(results[["fold", "n_train", "n_test", "mae", "r2"]].to_string(index=False))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    results.to_csv(cache_path, index=False)
    write_meta(cache_path, [data_path], settings, extra_files=(_CRABNET_OOD_PY,))
    print(f"Results cached to {cache_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=_DEFAULT_FILTERED_PATH)
    parser.add_argument("--threshold-ev", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--train-subset", choices=["dedup", "sizematched"], default=None,
                        help="replace each fold's TRAINING side with this MD-HIT arm "
                             "(test side unchanged); default = every training row")
    # Fan-out controls, out of the cache key so every shard lands on the same hash.
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument("--fold-index", type=int, default=None,
                     help="run only the Nth eligible fold; pass $SLURM_ARRAY_TASK_ID here "
                          "to fan the suite out as an array job")
    sel.add_argument("--only-folds", nargs="+", default=None,
                     help="run only these fold labels (e.g. lomo_Oxides)")
    parser.add_argument("--list-folds", action="store_true",
                        help="print index/label/sizes for every eligible fold and exit "
                             "(use to size an --array range)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="combine finished per-fold results into the results CSV; "
                             "writes nothing unless every eligible fold is present")
    parser.add_argument("--no-resume", action="store_true",
                        help="recompute folds that already have a cached result")
    args = parser.parse_args()

    run_crabnet_ood(
        data_path=args.data_file, threshold_ev=args.threshold_ev, seed=args.seed,
        epochs=args.epochs, force_cpu=args.force_cpu, max_folds=args.max_folds,
        train_subset=args.train_subset, fold_index=args.fold_index,
        only_folds=args.only_folds, resume=not args.no_resume,
        aggregate_only=args.aggregate_only, list_only=args.list_folds,
    )
