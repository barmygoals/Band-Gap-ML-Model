"""XGBoost through the full OOD suite and gap-range extrapolation on the pinned dataset,
reading the same folds FlexNet and CrabNet consume from the shared artifact
(model/folds.py) rather than rebuilding them.

Before 2026-07-27 it covered LOMO and extrapolation alone and rebuilt both recipes inline,
which is how it came to disagree with crabnet_ood about the extrapolation control.

Trees cannot predict above their maximum training label, so the extrapolation fold probes a
structural ceiling rather than a fitting failure.

Run: uv run python scripts/xgb_ood.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import write_meta
from config import PINNED_DATASET
from model.benchmark import _preprocess_fold
from model.training import prepare_features

XGB_PARAMS = dict(  # identical to scripts/xgb_baseline.py
    n_estimators=2000, learning_rate=0.05, max_depth=8, subsample=0.8,
    colsample_bytree=0.8, tree_method="hist", early_stopping_rounds=50, n_jobs=4,  # n_jobs pinned: see scripts/xgb_baseline.py
    eval_metric=["mae", "rmse"],  # rmse stays last to keep the stopping criterion unchanged, see scripts/xgb_baseline.py
)
THRESHOLD_EV = 4.0


def _fit_eval(X_tr, y_tr, X_te, y_te, seed: int) -> tuple[dict, np.ndarray, "object", dict]:
    from xgboost import XGBRegressor
    from model.crabnet_baseline import _metrics
    rng = np.random.default_rng(seed)
    val = rng.choice(len(X_tr), max(1, len(X_tr) // 10), replace=False)
    tr = np.setdiff1d(np.arange(len(X_tr)), val)
    m = XGBRegressor(random_state=seed, **XGB_PARAMS)
    # Val last, as early stopping watches the last eval_set (see xgb_baseline.py).
    t0 = time.perf_counter()
    m.fit(X_tr[tr], y_tr[tr], eval_set=[(X_tr[tr], y_tr[tr]), (X_tr[val], y_tr[val])], verbose=False)
    train_seconds = time.perf_counter() - t0
    # best_iteration recorded on the OOD rows too, the tree analogue of FlexNet's best_epoch.
    best_iter = getattr(m, "best_iteration", None)
    p = m.predict(X_te)
    # crabnet_baseline's _metrics, reused so all three models carry the same columns.
    metrics = {**_metrics(y_te, np.asarray(p, dtype="float64")),
               "max_pred": float(p.max()), "n_train": int(len(tr)), "n_val": int(len(val)),
               "train_seconds": train_seconds,
               "best_iteration": int(best_iter) if best_iter is not None else XGB_PARAMS["n_estimators"],
               "n_estimators": XGB_PARAMS["n_estimators"]}
    return metrics, p, m, m.evals_result()


def main(train_subset: str | None = None) -> None:
    X_df, y_s, formulas_s, _ = prepare_features(PINNED_DATASET)
    y = np.asarray(y_s, dtype="float64")
    formulas = np.asarray(formulas_s)
    X_raw = X_df.values
    # Non-default arms get a suffixed output, so the three cannot overwrite one another.
    # The bare xgb_ood.csv that RESULTS.md and master_summary.py read stays the full-data arm.
    suffix = f"_{train_subset}" if train_subset else ""
    model_dir = os.path.join("cache", "analysis", f"xgb_ood{suffix}_models")
    os.makedirs(model_dir, exist_ok=True)
    rows = []
    pred_frames = []
    imp_frames = []
    curve_frames = []

    def _run_fold(benchmark, fold, X_tr, X_te, tr_idx, te_idx, pipe):
        for seed in (42, 43, 44):
            r, p, m, ev = _fit_eval(X_tr, y[tr_idx], X_te, y[te_idx], seed)
            rows.append({"benchmark": benchmark, "fold": fold, "seed": seed,
                         "n_test": len(te_idx), **r})
            pred_frames.append(pd.DataFrame({
                "benchmark": benchmark, "fold": fold, "seed": seed, "row_idx": te_idx,
                "formula": formulas[te_idx], "y_true": y[te_idx], "y_pred": p,
            }))
            # Per-boosting-round curves (2026-08-14), so all three arms carry one.
            # n_trees is the analogue of epoch rather than the same unit, so they share no x-axis.
            for eval_key, dataset in (("validation_0", "train"), ("validation_1", "val")):
                for metric, series in ev.get(eval_key, {}).items():
                    curve_frames.append(pd.DataFrame({
                        "benchmark": benchmark, "fold": fold, "seed": seed,
                        "dataset": dataset, "n_trees": np.arange(1, len(series) + 1),
                        "metric": metric, "value": series,
                    }))
            imp_frames.append(pd.DataFrame({
                "benchmark": benchmark, "fold": fold, "seed": seed,
                "feature": pipe.feature_names, "importance": m.feature_importances_,
            }))
            m.save_model(os.path.join(model_dir, f"{benchmark}_{fold}_seed{seed}.json"))

    # Shared artifact, where formulas= asserts row ordering, as folds are row-index based.
    from model.folds import build_ood_folds, load_ood_folds
    from model.train_subsets import apply_subset, resolve_train_subset
    folds_dir = build_ood_folds(PINNED_DATASET, threshold_ev=THRESHOLD_EV)
    # Training side only, te_idx stays the artifact's.
    subsets = resolve_train_subset(train_subset, PINNED_DATASET, formulas,
                                   threshold_ev=THRESHOLD_EV)
    for split, fold, tr_idx, te_idx in load_ood_folds(folds_dir, formulas=formulas):
        tr_idx = apply_subset(subsets, split, fold, tr_idx)
        X_tr, X_te, pipe = _preprocess_fold(X_raw, y, tr_idx, te_idx, return_pipeline=True,
                                            feature_names=list(X_df.columns))
        _run_fold(split, fold, X_tr, X_te, tr_idx, te_idx, pipe)
        sub = rows[-3:]
        extra = ""
        if split == "extrapolation":
            extra = (f"  (max prediction {max(r_['max_pred'] for r_ in sub):.2f} eV; "
                     f"test min label {y[te_idx].min():.2f})")
        print(f"  {split}/{fold}: MAE {np.mean([r_['mae'] for r_ in sub]):.4f}"
              f"  n_test={len(te_idx)}{extra}")

    df = pd.DataFrame(rows)
    out = os.path.join("cache", "analysis", f"xgb_ood{suffix}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    write_meta(out, [PINNED_DATASET],
               {"xgb_params": XGB_PARAMS, "threshold_ev": THRESHOLD_EV, "seeds": [42, 43, 44],
                **({"train_subset": train_subset} if train_subset else {})},
               name=f"xgb_ood{suffix}")
    pd.concat(pred_frames, ignore_index=True).to_csv(out.replace(".csv", "_preds.csv"), index=False)
    pd.concat(imp_frames, ignore_index=True).to_csv(out.replace(".csv", "_importances.csv"), index=False)
    if curve_frames:
        pd.concat(curve_frames, ignore_index=True).to_csv(out.replace(".csv", "_valcurve.csv"), index=False)
    for b in df.benchmark.unique():
        sub = df[df.benchmark == b].groupby("fold")["mae"].mean()
        print(f"{b}: " + "  ".join(f"{k}={v:.4f}" for k, v in sub.items()))
    print(f"Written to {out} (+ _preds.csv, _importances.csv, _valcurve.csv, models under {model_dir})")


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser(description=__doc__)
    _p.add_argument("--train-subset", choices=["dedup", "sizematched"], default=None,
                    help="replace each fold's TRAINING side with this MD-HIT arm "
                         "(test side unchanged); default = every training row")
    main(**vars(_p.parse_args()))
