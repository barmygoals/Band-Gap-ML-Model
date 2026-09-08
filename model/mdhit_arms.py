"""MD-HIT in-distribution comparison, with three training arms scored against one fixed
test set.

The test set is carved out first and never de-duplicated, and the remaining pool becomes
three arms (full, dedup at `threshold`, and a size-matched random control), all scored on the
same locked rows. The earlier run_mdhit_comparison re-split per dataset, so each arm was
scored on different rows and the redundancy term was confounded (DECISIONS.md 2026-08-14).

Each test row is also tagged with whether it would survive de-duplication, which answers the
benchmark-inflation question without another training run.

Run: python -m model.mdhit_arms, or an "mdhit_arms" experiment_plan.yaml step.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cache import data_stem, get_cache_path, resolve_architecture, write_meta
from config import PINNED_DATASET
from data.preprocessing import _elmd_filter
from model.benchmark import _preprocess_fold, _train_fold
from model.crabnet_baseline import _harvest_curves, _load_crabnet, _metrics
from model.train_subsets import DEFAULT_SIMILARITY, DEFAULT_THRESHOLD, _rng_for
from model.training import (DEFAULT_ACTIVATIONS, DEFAULT_LAYER_WIDTHS, _NETWORK_PY,
                            _TRAINING_PY, prepare_features)

MODEL_ORDER = ("flexnet", "xgboost", "crabnet")

_MDHIT_ARMS_PY = os.path.abspath(__file__)
_BENCHMARK_PY = os.path.join(os.path.dirname(_MDHIT_ARMS_PY), "benchmark.py")

ARM_ORDER = ("full", "dedup", "sizematched")


def _arm_val_split(tr_idx: np.ndarray, seed: int, val_fraction: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Split an arm's training rows into (fit, val), carved from inside the arm rather than the
    full population, else the de-duplicated arm would be handed rows de-duplication had
    removed. Seeded by (seed) alone through default_rng rather than the torch or global numpy
    stream, so turning validation on cannot perturb any model's weight initialisation."""
    n_val = max(int(round(len(tr_idx) * val_fraction)), 1)
    order = np.random.default_rng(seed).permutation(len(tr_idx))
    return np.sort(tr_idx[order[n_val:]]), np.sort(tr_idx[order[:n_val]])


def _dedup_mask(formulas: np.ndarray, threshold: float, similarity: str,
                n_processes: int) -> np.ndarray:
    """Boolean mask over `formulas`, True where the row survives MD-HIT, taken row-level from a
    formula-level decision as polymorphs share a formula."""
    kept = _elmd_filter(pd.DataFrame({"formula": formulas}), threshold, "formula",
                        similarity, n_processes)
    return np.array([f in kept for f in formulas])


def _fit_xgboost(X_raw, y, tr_idx, test_idx, seed, feature_names, val_fraction):
    """XGBoost arm on the same rows FlexNet sees, early-stopped on a validation split carved
    from inside the arm. The pipeline is fit on the fit rows alone, following
    scripts/xgb_baseline.py, whose parameters and eval_metric order this reuses."""
    import time
    from xgboost import XGBRegressor
    from model.training import _fit_pipeline
    # Imported rather than copied, so the parameters and eval_metric order are pinned in one place.
    from scripts.xgb_baseline import XGB_PARAMS

    fit_idx, val_idx = _arm_val_split(tr_idx, seed, val_fraction)
    Xfit_t, pipe = _fit_pipeline(X_raw[fit_idx], y[fit_idx], feature_names=feature_names)
    Xval_t, Xte_t = pipe.transform(X_raw[val_idx]), pipe.transform(X_raw[test_idx])
    m = XGBRegressor(random_state=seed, **XGB_PARAMS)
    t0 = time.perf_counter()
    m.fit(Xfit_t, y[fit_idx], eval_set=[(Xfit_t, y[fit_idx]), (Xval_t, y[val_idx])], verbose=False)
    secs = time.perf_counter() - t0
    preds = np.asarray(m.predict(Xte_t), dtype="float64")
    best = getattr(m, "best_iteration", None)
    cost = {"train_seconds": secs, "n_fit_rows": int(len(fit_idx)),
            "best_iteration": int(best) if best is not None else XGB_PARAMS["n_estimators"],
            "n_estimators": XGB_PARAMS["n_estimators"]}
    return _metrics(y[test_idx], preds), preds, cost, None


def _fit_crabnet(formulas, y, tr_idx, test_idx, seed, val_fraction, epochs, force_cpu,
                 out_dir, tag):
    """CrabNet arm trained on the arm's formulas, validated from inside the arm, predicting the
    same locked test set. The split is the arm's own rather than phase 2's exported seed-42
    split, so these numbers are comparable to the other two models here rather than to phase
    2's CrabNet."""
    import time
    CrabNet = _load_crabnet()
    if CrabNet is None:
        return None, None, None, None
    import torch
    from model.crabnet_baseline import _cost as _crab_cost

    fit_idx, val_idx = _arm_val_split(tr_idx, seed, val_fraction)

    def _df(idx):
        return pd.DataFrame({"formula": formulas[idx], "target": y[idx]})

    torch.manual_seed(seed)
    np.random.seed(seed)
    kwargs = {"mat_prop": "band_gap", "losscurve": False, "learningcurve": False,
              "force_cpu": force_cpu, "verbose": False, "random_state": seed,
              "model_name": f"crabnet_mdhitarms_{tag}"}
    if epochs is not None:
        kwargs["epochs"] = epochs
    cb = CrabNet(**kwargs)
    t0 = time.perf_counter()
    cb.fit(_df(fit_idx), _df(val_idx))
    secs = time.perf_counter() - t0
    out = cb.predict(_df(test_idx), return_uncertainty=True)
    preds = np.asarray(out[0] if isinstance(out, tuple) else out, dtype="float64")
    _harvest_curves(cb, tag, out_dir)
    return _metrics(y[test_idx], preds), preds, _crab_cost(cb, secs, len(fit_idx)), None


def run_mdhit_arms(
    data_path: str = PINNED_DATASET,
    test_size: float = 0.15,
    split_seed: int = 42,
    threshold: float = DEFAULT_THRESHOLD,
    similarity: str = DEFAULT_SIMILARITY,
    n_processes: int = 10,
    layer_widths: list[int] | None = None,
    activations: list[str] | None = None,
    epochs: int = 200,
    lr: float = 1.4e-3,
    batch_size: int = 256,
    log_transform: bool = True,
    optimiser_name: str = "adamw",
    negative_gap_penalty_weight: float = 0.05,
    seeds: tuple[int, ...] = (42, 43, 44),
    val_checkpoint: bool = False,
    val_fraction: float = 0.1,
    models: tuple[str, ...] = MODEL_ORDER,
    crabnet_epochs: int | None = None,
    force_cpu: bool = False,
) -> pd.DataFrame:
    """One row per (arm, seed), every arm scored on the same locked test set.

    split_seed fixes the test carve-out and the size-matched draw and is not a training seed,
    as `seeds` varies initialisation within each arm, so arm-to-arm differences cannot be a
    split artefact.

    val_checkpoint carves validation from inside each arm's own pool, else the deduped arm
    would be handed rows de-duplication had removed.
    """
    if layer_widths is None:
        layer_widths = DEFAULT_LAYER_WIDTHS
    if activations is None:
        activations = DEFAULT_ACTIVATIONS
    layer_widths, activations, lr, optimiser_name = resolve_architecture(
        layer_widths, activations, lr, False, optimiser_name
    )
    settings = {
        "test_size": test_size, "split_seed": split_seed, "threshold": threshold,
        "similarity": similarity, "layer_widths": layer_widths, "activations": activations,
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "log_transform": log_transform, "optimiser_name": optimiser_name,
        "negative_gap_penalty_weight": negative_gap_penalty_weight, "seeds": list(seeds),
        "val_checkpoint": val_checkpoint, "val_fraction": val_fraction,
        # models is in the key, so a subset run is a different artifact rather than a partial one.
        "models": list(models), "crabnet_epochs": crabnet_epochs,
    }
    cache_path = get_cache_path("mdhit_arms", [data_path], settings,
                                extra_files=(_MDHIT_ARMS_PY, _NETWORK_PY, _BENCHMARK_PY,
                                             _TRAINING_PY), ext="csv")
    if os.path.exists(cache_path):
        print(f"Cache valid — loading MD-HIT arms from {cache_path}")
        return pd.read_csv(cache_path)

    import torch
    X_df, y_s, formulas_s, _ = prepare_features(data_path)
    y = np.asarray(y_s, dtype="float64")
    formulas = np.array(formulas_s.tolist())
    X_raw = X_df.values
    indices = np.arange(len(y))

    # _train_val_test_split's first cut at the same seed, so test_idx is the headline's held-out set.
    pool_idx, test_idx = train_test_split(indices, test_size=test_size,
                                          random_state=split_seed)
    pool_idx, test_idx = np.sort(pool_idx), np.sort(test_idx)
    print(f"MD-HIT arms on {data_path}: {len(y)} rows -> pool {len(pool_idx)} / "
          f"test {len(test_idx)} (locked, seed {split_seed})")

    print(f"De-duplicating the TRAINING POOL (t={threshold}, {similarity})...")
    dedup_idx = pool_idx[_dedup_mask(formulas[pool_idx], threshold, similarity, n_processes)]
    rng = _rng_for(split_seed, "mdhit_arms", "sizematched")
    sm_idx = np.sort(rng.choice(pool_idx, size=len(dedup_idx), replace=False))
    arms = {"full": pool_idx, "dedup": dedup_idx, "sizematched": sm_idx}
    for name in ARM_ORDER:
        print(f"  arm {name:<12} {len(arms[name]):>6} training rows "
              f"({len(arms[name]) / len(pool_idx):.1%} of pool)")

    # A labelling pass alone, never used for training, so inflation is answerable without retraining.
    print("Tagging test-set redundancy (labelling only, no effect on training)...")
    test_survivor = _dedup_mask(formulas[test_idx], threshold, similarity, n_processes)
    print(f"  {test_survivor.sum()}/{len(test_idx)} test rows survive de-duplication "
          f"({test_survivor.mean():.1%})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.dirname(cache_path)
    os.makedirs(out_dir, exist_ok=True)
    feature_names = list(X_df.columns)
    # Per-fit checkpointing (2026-08-14), else a wall-kill costs every finished fit.
    # Each (model, arm, seed) lands on disk as it finishes and is skipped on resume.
    fit_dir = cache_path.replace(".csv", "_fits")
    os.makedirs(fit_dir, exist_ok=True)

    def _fit_paths(model_name, arm, seed):
        stem = os.path.join(fit_dir, f"{model_name}_{arm}_seed{seed}")
        return stem + ".json", stem + "_preds.csv", stem + "_history.csv"

    rows: list[dict] = []
    pred_frames: list = []
    hist_frames: list = []
    for arm in ARM_ORDER:
        tr_idx = arms[arm]
        # Hoisted for FlexNet, which would otherwise refit an identical pipeline once per seed.
        X_tr, X_te = _preprocess_fold(X_raw, y, tr_idx, test_idx)
        for model_name in models:
            for seed in seeds:
                hist = None
                row_json, pred_csv, hist_csv = _fit_paths(model_name, arm, seed)
                if os.path.exists(row_json):
                    with open(row_json, encoding="utf-8") as fh:
                        rows.append(json.load(fh))
                    pred_frames.append(pd.read_csv(pred_csv))
                    if os.path.exists(hist_csv):
                        hist_frames.append(pd.read_csv(hist_csv))
                    print(f"  [{model_name:<8} {arm:<12} seed={seed}]  cached — skipping")
                    continue
                if model_name == "flexnet":
                    metrics, preds, extras = _train_fold(
                        X_tr, y[tr_idx], X_te, y[test_idx], device,
                        layer_widths=layer_widths, activations=activations,
                        epochs=epochs, lr=lr, batch_size=batch_size,
                        log_transform=log_transform, optimiser_name=optimiser_name,
                        random_state=seed,
                        negative_gap_penalty_weight=negative_gap_penalty_weight,
                        val_checkpoint=val_checkpoint, val_fraction=val_fraction,
                    )
                    cost, hist = extras["cost"], extras["history"]
                elif model_name == "xgboost":
                    metrics, preds, cost, _ = _fit_xgboost(
                        X_raw, y, tr_idx, test_idx, seed, feature_names, val_fraction)
                elif model_name == "crabnet":
                    metrics, preds, cost, _ = _fit_crabnet(
                        formulas, y, tr_idx, test_idx, seed, val_fraction, crabnet_epochs,
                        force_cpu, out_dir, f"{data_stem(data_path)}_{arm}_seed{seed}")
                    if metrics is None:      # crabnet not installed, so skip rather than fail
                        print(f"  [{arm}] crabnet unavailable, skipping")
                        continue
                else:
                    raise ValueError(f"unknown model {model_name!r}; expected {MODEL_ORDER}")

                # MAE on the non-redundant test rows alone, the inflation number rather than a second model.
                mae_nr = float(np.abs(preds[test_survivor] - y[test_idx][test_survivor]).mean())
                row = {"model": model_name, "arm": arm, "seed": seed,
                       "n_train": len(tr_idx), "n_test": len(test_idx),
                       **metrics, "mae_test_nonredundant": mae_nr, **cost}
                rows.append(row)
                print(f"  [{model_name:<8} {arm:<12} seed={seed}]  MAE={metrics['mae']:.4f}  "
                      f"(non-redundant {mae_nr:.4f})  R²={metrics['r2']:.4f}")
                pred_df = pd.DataFrame({
                    "model": model_name, "arm": arm, "seed": seed, "row_idx": test_idx,
                    "formula": formulas[test_idx], "y_true": y[test_idx], "y_pred": preds,
                    "test_dedup_survivor": test_survivor,
                })
                pred_frames.append(pred_df)
                hist_df = None
                if hist is not None:
                    frame = {"model": model_name, "arm": arm, "seed": seed,
                             "epoch": np.arange(1, len(hist["train_loss"]) + 1),
                             "train_loss": hist["train_loss"],
                             "train_rmse_fit": hist["train_rmse_fit"]}
                    for key in ("val_mae", "val_rmse"):
                        if key in hist:
                            frame[key] = hist[key]
                    hist_df = pd.DataFrame(frame)
                    hist_frames.append(hist_df)
                # Checkpoint last, as the JSON is the resume gate and must not outlive its own outputs.
                pred_df.to_csv(pred_csv, index=False)
                if hist_df is not None:
                    hist_df.to_csv(hist_csv, index=False)
                with open(row_json, "w", encoding="utf-8") as fh:
                    json.dump(row, fh)

    results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    results.to_csv(cache_path, index=False)
    write_meta(cache_path, [data_path], settings,
               extra_files=(_MDHIT_ARMS_PY, _NETWORK_PY, _BENCHMARK_PY, _TRAINING_PY))

    pd.concat(pred_frames, ignore_index=True).to_csv(
        cache_path.replace(".csv", "_preds.csv"), index=False)
    pd.concat(hist_frames, ignore_index=True).to_csv(
        cache_path.replace(".csv", "_history.csv"), index=False)
    # Membership, else the arms cannot be reconstructed from the results alone.
    member = [pd.DataFrame({"role": "test", "row_idx": test_idx,
                            "formula": formulas[test_idx],
                            "test_dedup_survivor": test_survivor})]
    for arm in ARM_ORDER:
        member.append(pd.DataFrame({"role": f"train_{arm}", "row_idx": arms[arm],
                                    "formula": formulas[arms[arm]],
                                    "test_dedup_survivor": False}))
    pd.concat(member, ignore_index=True).to_csv(
        cache_path.replace(".csv", "_membership.csv"), index=False)

    print(f"\n=== MD-HIT arms — identical locked test set ({len(test_idx)} rows) ===")
    for model_name in models:
        sub = results[results.model == model_name]
        if not len(sub):
            continue
        print(f"\n-- {model_name} --")
        agg = sub.groupby("arm")[["mae", "mae_test_nonredundant"]].agg(["mean", "std"])
        print(agg.reindex([a for a in ARM_ORDER if a in set(sub.arm)]).to_string())
        _paired(sub, "full", "dedup", "total effect  (full - dedup)")
        _paired(sub, "full", "sizematched", "volume        (full - control)")
        _paired(sub, "sizematched", "dedup", "redundancy    (control - dedup)")
    print(f"\nResults cached to {cache_path}")
    return results


def _paired(results: pd.DataFrame, a: str, b: str, label: str) -> None:
    """Seed-paired delta between two arms, tighter than a difference of means as every arm ran the same seeds on the same test set."""
    pa = results[results.arm == a].set_index("seed")["mae"]
    pb = results[results.arm == b].set_index("seed")["mae"]
    common = sorted(set(pa.index) & set(pb.index))
    if not common:
        return
    d = np.array([pa[s] - pb[s] for s in common])
    print(f"  {label}: {d.mean():+.4f} +- {d.std(ddof=1) if len(d) > 1 else float('nan'):.4f} "
          f"({len(d)} seeds)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default=PINNED_DATASET)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1.4e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--val-checkpoint", action="store_true")
    p.add_argument("--np", type=int, default=10, dest="n_processes")
    p.add_argument("--models", nargs="+", default=list(MODEL_ORDER),
                   choices=list(MODEL_ORDER),
                   help="which model classes to run (default: all three)")
    p.add_argument("--crabnet-epochs", type=int, default=None,
                   help="override CrabNet's epoch ceiling (smoke tests only)")
    p.add_argument("--force-cpu", action="store_true")
    a = p.parse_args()
    run_mdhit_arms(data_path=a.data_path, test_size=a.test_size, split_seed=a.split_seed,
                   threshold=a.threshold, epochs=a.epochs, lr=a.lr, seeds=tuple(a.seeds),
                   val_checkpoint=a.val_checkpoint, n_processes=a.n_processes,
                   models=tuple(a.models), crabnet_epochs=a.crabnet_epochs,
                   force_cpu=a.force_cpu)
