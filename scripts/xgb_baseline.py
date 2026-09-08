"""XGBoost baseline on the pinned dataset at the headline 70/15/15 split, the canonical
XGB configuration RESULTS.md quotes. Features and preprocessing are identical to FlexNet's,
and it early-stops on the val split.

feature_tier selects the descriptor set, so this doubles as the XGBoost arm of the phase-1
physics ladder. CrabNet is absent from the ladder, as it builds its own feature space from
the formula (DECISIONS.md 2026-07-27).

Non-standard tiers write to a tier-suffixed path, so rungs cannot overwrite one another.

Run: uv run python scripts/xgb_baseline.py [--seeds 42 43 44] [--feature-tier physics]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import write_meta
from config import PINNED_DATASET
from model.training import prepare_features, _train_val_test_split, _fit_pipeline

XGB_PARAMS = dict(
    n_estimators=2000, learning_rate=0.05, max_depth=8, subsample=0.8,
    colsample_bytree=0.8, tree_method="hist", early_stopping_rounds=50,
    # XGBoost early-stops on the last eval_set's last metric, so rmse stays last (2026-08-14).
    # mae is logged alongside, so the tree curve is comparable to FlexNet's and CrabNet's.
    eval_metric=["mae", "rmse"],
    # n_jobs pinned to --cpus-per-task, hist reduction being thread-order dependent (2026-07-28).
    # Keep it in step with the SLURM allocation if that changes.
    n_jobs=4,
)


def main(seeds: list[int], feature_tier: str = "standard", early_stopping: bool = True,
         data_path: str = PINNED_DATASET, n_estimators: int | None = None) -> None:
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X_df, y_s, formulas_s, _ = prepare_features(data_path, feature_tier=feature_tier)
    y = np.asarray(y_s, dtype="float64")
    formulas = np.asarray(formulas_s)
    # Empty suffix for the pinned dataset at the standard tier, so pre-ladder paths hold.
    # Every other combination gets a suffix, so arms cannot overwrite one another.
    suffix = "" if feature_tier == "standard" else f"_{feature_tier}"
    if data_path != PINNED_DATASET:
        suffix += "_" + os.path.splitext(os.path.basename(data_path))[0]
    params = dict(XGB_PARAMS)
    if n_estimators is not None and n_estimators != params["n_estimators"]:
        params["n_estimators"] = n_estimators
        suffix += f"_{n_estimators}trees"
    if not early_stopping:
        params.pop("early_stopping_rounds")
        suffix += "_noearlystop"
    model_dir = os.path.join("cache", "analysis", f"xgb_baseline{suffix}_models")
    os.makedirs(model_dir, exist_ok=True)
    rows = []
    pred_frames = []
    imp_frames = []
    curve_frames = []
    for seed in seeds:
        Xtr, Xval, Xte, ytr, yval, yte, _, idx_test = _train_val_test_split(X_df.values, y, 0.15, 0.15, seed)
        Xtr_t, pipe = _fit_pipeline(Xtr, ytr, feature_names=list(X_df.columns))
        Xval_t, Xte_t = pipe.transform(Xval), pipe.transform(Xte)
        model = XGBRegressor(random_state=seed, **params)
        # eval_set is passed either way, so the val curve is recorded with or without stopping.
        # Train is validation_0 (2026-08-14), so the curve shows the train/val gap.
        # Val stays last, as early stopping watches the last eval_set.
        t0 = time.perf_counter()
        model.fit(Xtr_t, ytr, eval_set=[(Xtr_t, ytr), (Xval_t, yval)], verbose=False)
        train_seconds = time.perf_counter() - t0
        # Persist the curve, FlexNet's per-epoch history analogue (DECISIONS.md 2026-07-27).
        ev = model.evals_result()
        for eval_key, dataset in (("validation_0", "train"), ("validation_1", "val")):
            for metric, series in ev.get(eval_key, {}).items():
                curve_frames.append(pd.DataFrame({
                    "seed": seed, "dataset": dataset,
                    "n_trees": np.arange(1, len(series) + 1),
                    "metric": metric, "value": series,
                }))
        pred = model.predict(Xte_t)
        best_iter = getattr(model, "best_iteration", None)
        row = {
            "seed": seed, "mae": mean_absolute_error(yte, pred),
            "rmse": float(np.sqrt(mean_squared_error(yte, pred))), "r2": r2_score(yte, pred),
            "best_iteration": int(best_iter) if best_iter is not None else params["n_estimators"],
            # Cost record (2026-08-14), the same fields FlexNet's and CrabNet's arms write.
            # n_estimators is the ceiling, best_iteration what predict() uses.
            "train_seconds": train_seconds,
            "n_estimators": params["n_estimators"],
            "n_fit_rows": int(len(ytr)),
        }
        rows.append(row)
        pred_frames.append(pd.DataFrame({
            "seed": seed, "row_idx": idx_test, "formula": formulas[idx_test],
            "y_true": yte, "y_pred": pred,
        }))
        imp_frames.append(pd.DataFrame({
            "seed": seed, "feature": pipe.feature_names, "importance": model.feature_importances_,
        }))
        model.save_model(os.path.join(model_dir, f"seed{seed}.json"))
        print(f"seed {seed}: MAE={row['mae']:.4f}  RMSE={row['rmse']:.4f}  R2={row['r2']:.4f}")
    df = pd.DataFrame(rows)
    stop = "early-stopped" if early_stopping else "all trees"
    print(f"XGB baseline [{feature_tier}, {stop}]: {df.mae.mean():.4f} +- {df.mae.std(ddof=1):.4f} "
          f"({len(seeds)} seeds, {X_df.shape[1]} raw features, "
          f"mean {df.best_iteration.mean():.0f} trees)")
    out = os.path.join("cache", "analysis", f"xgb_baseline{suffix}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    # data_path rather than PINNED_DATASET, as the MD-HIT arms train on other datasets.
    write_meta(out, [data_path],
               {"seeds": seeds, "feature_tier": feature_tier,
                "early_stopping": early_stopping, **params},
               name=f"xgb_baseline{suffix}")
    pd.concat(pred_frames, ignore_index=True).to_csv(out.replace(".csv", "_preds.csv"), index=False)
    pd.concat(imp_frames, ignore_index=True).to_csv(out.replace(".csv", "_importances.csv"), index=False)
    if curve_frames:
        pd.concat(curve_frames, ignore_index=True).to_csv(out.replace(".csv", "_valcurve.csv"), index=False)
    print(f"Written to {out} (+ _preds.csv, _importances.csv, _valcurve.csv, models under {model_dir})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--feature-tier", default="standard",
                        help="descriptor set: fractions | literature | standard | physics | physics-alt")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="keep all n_estimators trees (the FlexNet-last-epoch analogue)")
    parser.add_argument("--data-path", default=PINNED_DATASET,
                        help="dataset to train on; non-pinned paths get a suffixed output")
    parser.add_argument("--n-estimators", type=int, default=None,
                        help="override the tree budget (default 2000)")
    args = parser.parse_args()
    main(args.seeds, feature_tier=args.feature_tier,
         early_stopping=not args.no_early_stopping, data_path=args.data_path,
         n_estimators=args.n_estimators)
