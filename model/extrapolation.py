"""Target-range extrapolation test, training on Eg < threshold and evaluating on the
held-out wide-gap tail.

The OOD suite probes chemical extrapolation but never value extrapolation, so this separates
chemistry novelty from target-range novelty.

Three folds, each trained across `seeds`:
  extrapolation    train Eg < threshold, test Eg >= threshold
  random_control   the same sizes drawn at random, controlling for training-set size
  inrange_control  train plus half the wide-gap rows, test the other half

Absolute errors scale with gap magnitude even in-distribution, so the corrected penalty
subtracts inrange_control's MAE rather than random_control's.

Run: python -m model.extrapolation, or a "gap_extrapolation" plan step.
"""
import os

import joblib
import numpy as np
import pandas as pd
import torch

from cache import get_cache_path, resolve_architecture, write_meta
from data.preprocessing import default_filtered_data_path

from config import PINNED_DATASET as _DEFAULT_FILTERED_PATH
from model.benchmark import _BENCHMARK_PY, _NETWORK_PY, _preprocess_fold, _train_fold
from model.training import DEFAULT_ACTIVATIONS, DEFAULT_LAYER_WIDTHS, prepare_features

_EXTRAPOLATION_PY = os.path.abspath(__file__)


def gap_extrapolation_folds(
    y: np.ndarray,
    threshold_ev: float = 4.0,
    seed: int = 42,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The three (fold_name, train_idx, test_idx) folds of the value-extrapolation test,
    extracted 2026-07-27 so there is one definition. The rng is consumed sequentially, so
    reordering these calls would change fold membership that has already been published, and y
    has to be float32 to reproduce it."""
    y = np.asarray(y, dtype="float32")
    indices = np.arange(len(y))
    wide = y >= threshold_ev
    extrap_train, extrap_test = indices[~wide], indices[wide]
    if len(extrap_test) < 10:
        raise ValueError(
            f"Only {len(extrap_test)} rows have Eg >= {threshold_ev} eV -- "
            "lower threshold_ev or use a larger dataset variant."
        )
    # Size-matched random control at identical train/test counts, with random membership
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    ctrl_test = shuffled[: len(extrap_test)]
    ctrl_train = shuffled[len(extrap_test): len(extrap_test) + len(extrap_train)]
    # In-range control, half the wide-gap rows training and half tested, for this label range.
    wide_idx = rng.permutation(extrap_test)
    half = len(wide_idx) // 2
    return [
        ("extrapolation", extrap_train, extrap_test),
        ("random_control", ctrl_train, ctrl_test),
        ("inrange_control", np.concatenate([extrap_train, wide_idx[:half]]), wide_idx[half:]),
    ]


def run_gap_extrapolation(
    data_path: str = _DEFAULT_FILTERED_PATH,
    threshold_ev: float = 4.0,
    layer_widths: list[int] | None = None,
    activations: list[str] | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    use_best_found: bool = False,
    log_transform: bool = False,
    optimiser_name: str = "adam",
    seeds: tuple[int, ...] = (42, 43, 44),
    negative_gap_penalty_weight: float = 0.05,
    val_checkpoint: bool = False,
    val_fraction: float = 0.1,
    train_subset: str | None = None,
) -> pd.DataFrame:
    """One row per (fold, seed) over the extrapolation, random_control and inrange_control
    folds, cached under cache/extrapolation_results/.

    The corrected penalty uses inrange_control rather than random_control, as absolute errors
    scale with gap magnitude even in-distribution. Parameters match run_benchmark()'s.
    """
    if layer_widths is None:
        layer_widths = DEFAULT_LAYER_WIDTHS
    if activations is None:
        activations = DEFAULT_ACTIVATIONS
    layer_widths, activations, lr, optimiser_name = resolve_architecture(
        layer_widths, activations, lr, use_best_found, optimiser_name
    )
    settings = {
        "threshold_ev": threshold_ev,
        "layer_widths": layer_widths, "activations": activations, "epochs": epochs, "lr": lr,
        "batch_size": batch_size, "log_transform": log_transform,
        "optimiser_name": optimiser_name, "seeds": list(seeds),
        "negative_gap_penalty_weight": negative_gap_penalty_weight,
    }
    # Recorded only when on, so existing fixed-epoch entries keep their hash.
    if val_checkpoint:
        settings["val_checkpoint"] = True
        settings["val_fraction"] = val_fraction
    if train_subset:
        settings["train_subset"] = train_subset
    extra = (_EXTRAPOLATION_PY, _NETWORK_PY, _BENCHMARK_PY)
    cache_path = get_cache_path("extrapolation_results", [data_path], settings, extra_files=extra, ext="csv")
    if os.path.exists(cache_path):
        print(f"Cache valid — loading extrapolation results from {cache_path}")
        return pd.read_csv(cache_path)

    X_df, y_s, formulas_s, _ = prepare_features(data_path)
    formulas = np.array(formulas_s.tolist())  # for the per-material predictions file
    y = np.asarray(y_s, dtype="float32")
    X_raw = X_df.values
    indices = np.arange(len(y))

    # The one shared constructor, where seeds[0] reproduces the historical rng draw.
    folds = gap_extrapolation_folds(y, threshold_ev, seed=seeds[0])
    # Training side only, test sides stay the artifact's.
    if train_subset:
        from model.train_subsets import apply_subset, resolve_train_subset
        _subs = resolve_train_subset(train_subset, data_path, formulas, threshold_ev=threshold_ev)
        folds = [(name, apply_subset(_subs, "extrapolation", name, tr), te)
                 for name, tr, te in folds]
    extrap_train, extrap_test = folds[0][1], folds[0][2]
    print(
        f"Extrapolation split at {threshold_ev} eV: train {len(extrap_train)} (Eg < t), "
        f"test {len(extrap_test)} (Eg >= t, {len(extrap_test) / len(y):.1%} of rows)"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = cache_path.replace(".csv", "_models")
    os.makedirs(model_dir, exist_ok=True)
    rows: list[dict] = []
    pred_frames: list = []
    hist_frames: list = []
    membership_frames: list = []
    for fold_name, tr_idx, te_idx in folds:
        X_tr, X_te, fold_pipeline = _preprocess_fold(
            X_raw, y, tr_idx, te_idx, return_pipeline=True, feature_names=list(X_df.columns)
        )
        y_tr, y_te = y[tr_idx], y[te_idx]
        joblib.dump(fold_pipeline, os.path.join(model_dir, f"{fold_name}_pipeline.joblib"))
        # Train-side membership, as a saved fold model is reusable only with its fit rows.
        membership_frames.append(pd.DataFrame({
            "fold": fold_name, "role": "train", "row_idx": np.asarray(tr_idx),
        }))
        for seed in seeds:
            metrics, preds, extras = _train_fold(
                X_tr, y_tr, X_te, y_te, device,
                layer_widths=layer_widths, activations=activations,
                epochs=epochs, lr=lr, batch_size=batch_size,
                log_transform=log_transform, optimiser_name=optimiser_name,
                random_state=seed,
                negative_gap_penalty_weight=negative_gap_penalty_weight,
                val_checkpoint=val_checkpoint,
                val_fraction=val_fraction,
            )
            print(f"  [{fold_name}] seed={seed}  MAE={metrics['mae']:.4f} eV  R²={metrics['r2']:.4f}"
                  f"  ({extras['cost']['train_seconds']:.1f}s)")
            rows.append({"fold": fold_name, "seed": seed,
                         "n_train": len(tr_idx), "n_test": len(te_idx), **metrics,
                         **extras["cost"]})
            pred_frames.append(pd.DataFrame({
                "fold": fold_name, "seed": seed, "row_idx": np.asarray(te_idx),
                "formula": formulas[te_idx], "y_true": y_te, "y_pred": preds,
            }))
            h = extras["history"]
            hist_frame = {
                "fold": fold_name, "seed": seed,
                "epoch": np.arange(1, len(h["train_loss"]) + 1),
                "train_loss": h["train_loss"], "train_rmse_fit": h["train_rmse_fit"],
            }
            for key in ("val_mae", "val_rmse"):  # present only under val_checkpoint
                if key in h:
                    hist_frame[key] = h[key]
            hist_frames.append(pd.DataFrame(hist_frame))
            torch.save(extras["model"].state_dict(),
                       os.path.join(model_dir, f"{fold_name}_seed{seed}.pt"))

    results = pd.DataFrame(rows)
    means = results.groupby("fold")["mae"].agg(["mean", "std"])
    print("\n=== Gap-range extrapolation summary ===")
    print(means.to_string())
    summary = means.copy()
    if {"extrapolation", "inrange_control"} <= set(means.index):
        penalty = means.loc["extrapolation", "mean"] - means.loc["inrange_control", "mean"]
        print(f"Extrapolation penalty (extrap MAE - in-range wide-gap MAE): {penalty:.4f} eV")
        summary.loc["penalty_vs_inrange"] = [penalty, float("nan")]
    if {"extrapolation", "random_control"} <= set(means.index):
        naive = means.loc["extrapolation", "mean"] - means.loc["random_control", "mean"]
        print(f"(Naive penalty vs random control, conflates label magnitude: {naive:.4f} eV)")
        summary.loc["penalty_vs_random"] = [naive, float("nan")]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    results.to_csv(cache_path, index=False)
    write_meta(cache_path, [data_path], settings, extra_files=extra)
    print(f"Results cached to {cache_path}")

    preds_path = cache_path.replace(".csv", "_preds.csv")
    pd.concat(pred_frames, ignore_index=True).to_csv(preds_path, index=False)
    print(f"Per-material predictions cached to {preds_path}")

    summary.to_csv(cache_path.replace(".csv", "_summary.csv"))
    pd.concat(hist_frames, ignore_index=True).to_csv(cache_path.replace(".csv", "_history.csv"), index=False)
    pd.concat(membership_frames, ignore_index=True).to_csv(cache_path.replace(".csv", "_membership.csv"), index=False)
    print(f"Summary, per-epoch curves, membership + fold models saved beside {cache_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=_DEFAULT_FILTERED_PATH)
    parser.add_argument("--threshold-ev", type=float, default=4.0)
    parser.add_argument("--layer-widths", type=int, nargs="+", default=None)
    parser.add_argument("--activations", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--optimiser", default="adam", dest="optimiser_name")
    parser.add_argument("--log-transform", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    run_gap_extrapolation(
        data_path=args.data_file, threshold_ev=args.threshold_ev,
        layer_widths=args.layer_widths, activations=args.activations,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        optimiser_name=args.optimiser_name, log_transform=args.log_transform,
        seeds=tuple(args.seeds),
    )
