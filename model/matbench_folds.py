"""Matbench mp_gap evaluation from the cached official fold CSVs, needing no `matbench`
package, which is no longer installable here as it pins a scikit-learn this project cannot
satisfy.

The fold CSVs are matbench's official partitions, so training on them reproduces the split
mechanics exactly. It is not a leaderboard submission, as there is no task.record()
validation, and that caveat belongs wherever these numbers are quoted.

It does not modify model/benchmark.py, whose content keys the OOD caches.

Run: python -m model.matbench_folds, or a "matbench_from_folds" plan step.
"""
import os

import joblib
import numpy as np
import pandas as pd
import torch

from cache import CACHE_DIR, get_cache_path, resolve_architecture, write_meta
from features.descriptors import build_nn_features
from model.benchmark import _BENCHMARK_PY, _NETWORK_PY, _preprocess_fold, _train_fold
from model.training import DEFAULT_ACTIVATIONS, DEFAULT_LAYER_WIDTHS

_MATBENCH_FOLDS_PY = os.path.abspath(__file__)
_DEFAULT_FOLD_DIR = os.path.join(CACHE_DIR, "matbench_folds")


def write_fold_union_csv(fold_idx: int, train_df: pd.DataFrame,
                         test_df: pd.DataFrame) -> str:
    """Write once and return the merged train-then-test CSV a matbench fold is featurised
    against, whose path defines the fold's feature cache key, so anything pre-building that
    matrix must write a byte-identical file (DECISIONS.md 2026-08-19)."""
    union_csv = os.path.join(CACHE_DIR, "analysis", f"matbench_fold{fold_idx}_union.csv")
    if not os.path.exists(union_csv):
        os.makedirs(os.path.dirname(union_csv), exist_ok=True)
        pd.concat([train_df, test_df], ignore_index=True).to_csv(union_csv, index=False)
    return union_csv


def run_matbench_from_folds(
    fold_dir: str = _DEFAULT_FOLD_DIR,
    layer_widths: list[int] | None = None,
    activations: list[str] | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    use_best_found: bool = False,
    log_transform: bool = False,
    optimiser_name: str = "adam",
    random_state: int = 42,
    negative_gap_penalty_weight: float = 0.05,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Train and evaluate FlexNet on each cached matbench fold, cached under
    cache/matbench_mp_gap_folds/.

    Returns per-fold metrics and their mean. Parameters match run_matbench_mp_gap()'s,
    including the deprecated use_best_found, where True raises. n_folds below 5 is for smoke
    tests alone.
    """
    if layer_widths is None:
        layer_widths = DEFAULT_LAYER_WIDTHS
    if activations is None:
        activations = DEFAULT_ACTIVATIONS
    layer_widths, activations, lr, optimiser_name = resolve_architecture(
        layer_widths, activations, lr, use_best_found, optimiser_name
    )

    fold_paths = []
    for i in range(n_folds):
        train_csv = os.path.join(fold_dir, f"matbench_fold{i}_train.csv")
        test_csv = os.path.join(fold_dir, f"matbench_fold{i}_test.csv")
        if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
            raise FileNotFoundError(
                f"Missing cached matbench fold CSVs for fold {i} under {fold_dir} -- "
                "these were written by the 2026-07-01 run_matbench_mp_gap() run and "
                "cannot be regenerated without the (uninstallable) matbench package."
            )
        fold_paths.append((train_csv, test_csv))

    settings = {
        "layer_widths": layer_widths, "activations": activations, "epochs": epochs, "lr": lr,
        "batch_size": batch_size, "log_transform": log_transform,
        "optimiser_name": optimiser_name, "random_state": random_state,
        "negative_gap_penalty_weight": negative_gap_penalty_weight, "n_folds": n_folds,
    }
    data_files = [p for pair in fold_paths for p in pair]
    cache_path = get_cache_path(
        "matbench_mp_gap_folds", data_files, settings,
        extra_files=(_MATBENCH_FOLDS_PY, _NETWORK_PY, _BENCHMARK_PY), ext="csv",
    )
    if os.path.exists(cache_path):
        print(f"Cache valid — loading matbench-from-folds results from {cache_path}")
        return pd.read_csv(cache_path)

    from pymatgen.core import Composition

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Matbench mp_gap (cached official folds)  |  device={device}")
    model_dir = cache_path.replace(".csv", "_models")
    os.makedirs(model_dir, exist_ok=True)
    rows: list[dict] = []
    pred_frames: list = []
    hist_frames: list = []

    for i, (train_csv, test_csv) in enumerate(fold_paths):
        print(f"\n=== Fold {i} ===")
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        train_comps = [Composition(f) for f in train_df["formula_pretty"]]
        test_comps = [Composition(f) for f in test_df["formula_pretty"]]
        y_train = np.asarray(train_df["band_gap"], dtype="float32")
        y_test = np.asarray(test_df["band_gap"], dtype="float32")

        print(f"  Computing features for {len(train_comps)} train + {len(test_comps)} test compositions...")
        # One matrix per fold, else each file gets its own element set and vstacking mispairs.
        # Not leakage, as which elements exist is a property of the compositions, not of a label.
        # The pipeline still fits on the train rows alone below.
        # write_fold_union_csv() keeps the cache key shared with scripts/prebuild_features.py.
        union_csv = write_fold_union_csv(i, train_df, test_df)
        X_combined_df = build_nn_features(train_comps + test_comps, union_csv)
        n_tr = len(train_comps)
        X_combined = X_combined_df.values
        y_combined = np.concatenate([y_train, y_test])
        X_tr, X_te, fold_pipeline = _preprocess_fold(
            X_combined, y_combined, np.arange(n_tr), np.arange(n_tr, n_tr + len(test_comps)),
            return_pipeline=True, feature_names=list(X_combined_df.columns),
        )
        joblib.dump(fold_pipeline, os.path.join(model_dir, f"fold{i}_pipeline.joblib"))

        print(f"  Training: {len(y_train)} samples  |  {X_tr.shape[1]} features")
        metrics, preds, extras = _train_fold(
            X_tr, y_train, X_te, y_test, device,
            layer_widths=layer_widths, activations=activations,
            epochs=epochs, lr=lr, batch_size=batch_size,
            log_transform=log_transform, optimiser_name=optimiser_name,
            random_state=random_state,
            negative_gap_penalty_weight=negative_gap_penalty_weight,
        )
        print(f"  Fold {i}: MAE={metrics['mae']:.4f} eV  R²={metrics['r2']:.4f}  "
              f"(nonmetals-only MAE={metrics['mae_nonmetals']:.4f})")
        rows.append({"fold": i, "n_train": len(y_train), "n_test": len(y_test), **metrics})
        pred_frames.append(pd.DataFrame({
            "fold": i, "seed": random_state,
            "formula": [str(c.reduced_formula) for c in test_comps],
            "y_true": y_test, "y_pred": preds,
        }))
        h = extras["history"]
        hist_frames.append(pd.DataFrame({
            "fold": i, "seed": random_state,
            "epoch": np.arange(1, len(h["train_loss"]) + 1),
            "train_loss": h["train_loss"], "train_rmse_fit": h["train_rmse_fit"],
        }))
        torch.save(extras["model"].state_dict(),
                   os.path.join(model_dir, f"fold{i}_seed{random_state}.pt"))

    results = pd.DataFrame(rows)
    summary = results[["mae", "rmse", "r2", "mrae", "mae_nonmetals"]].mean()
    print("\n=== Matbench mp_gap (cached folds) — mean across folds ===")
    print(f"  MAE  = {summary['mae']:.4f} eV   (nonmetals-only: {summary['mae_nonmetals']:.4f} eV)")
    print(f"  RMSE = {summary['rmse']:.4f} eV   R² = {summary['r2']:.4f}   MRAE = {summary['mrae']:.4f}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    results.to_csv(cache_path, index=False)
    write_meta(cache_path, data_files, settings, extra_files=(_MATBENCH_FOLDS_PY, _NETWORK_PY, _BENCHMARK_PY))
    print(f"Results cached to {cache_path}")

    preds_path = cache_path.replace(".csv", "_preds.csv")
    pd.concat(pred_frames, ignore_index=True).to_csv(preds_path, index=False)
    print(f"Per-material predictions cached to {preds_path}")

    summary.to_frame("mean_across_folds").to_csv(cache_path.replace(".csv", "_summary.csv"))
    pd.concat(hist_frames, ignore_index=True).to_csv(cache_path.replace(".csv", "_history.csv"), index=False)
    print(f"Summary, per-epoch curves + fold models saved beside {cache_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", default=_DEFAULT_FOLD_DIR)
    parser.add_argument("--layer-widths", type=int, nargs="+", default=None)
    parser.add_argument("--activations", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--optimiser", default="adam", dest="optimiser_name")
    parser.add_argument("--log-transform", action="store_true")
    parser.add_argument("--seed", type=int, default=42, dest="random_state")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    run_matbench_from_folds(
        fold_dir=args.fold_dir, layer_widths=args.layer_widths, activations=args.activations,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        log_transform=args.log_transform, optimiser_name=args.optimiser_name,
        random_state=args.random_state, n_folds=args.n_folds,
    )
