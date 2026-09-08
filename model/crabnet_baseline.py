"""CrabNet baseline (Wang, Kauwe, Murdock & Sparks 2021, npj Comput. Mater. 7:77), a
compositionally-restricted attention transformer trained on the same compositions and
partitions as FlexNet through model.export_splits, so any accuracy difference is
architecture rather than the split.

Not hyperparameter-searched, as the comparison is an Optuna-tuned FlexNet against a
published-default CrabNet, which if anything understates CrabNet. Where the `crabnet` package
is missing it prints instructions and returns None rather than failing a plan.

Run: python -m model.crabnet_baseline, or a "crabnet_baseline" plan step.
"""
import hashlib
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from cache import data_stem, get_cache_path, write_meta
from data.preprocessing import default_filtered_data_path

from config import PINNED_DATASET as _DEFAULT_FILTERED_PATH
from model.export_splits import export_splits

_CRABNET_BASELINE_PY = os.path.abspath(__file__)

# crabnet writes models/trained_models/NAME.pth relative to the CWD and reads from there.
# Passing full paths breaks both directions, so everything here uses bare _network_name().
_NETWORK_DIR = os.path.join("models", "trained_models")


def _network_name(data_path: str, seed: int) -> str:
    return f"crabnet_{data_stem(data_path)}_seed{seed}"


# Checkpoint integrity (2026-07-23), as crabnet's fit() auto-saves under model_name.
# A stray fit can overwrite a real checkpoint unreported, which broke the July transfer runs.
# The .pth.md5 sidecar lets every consumer check the file is the one the baseline saved.

def _checkpoint_md5(weights_path: str) -> str:
    with open(weights_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def write_checkpoint_md5(name: str) -> None:
    """Fingerprint NAME.pth into NAME.pth.md5, called immediately after a deliberate save_network() and a no-op where the save failed."""
    weights_path = os.path.join(_NETWORK_DIR, f"{name}.pth")
    if os.path.exists(weights_path):
        md5 = _checkpoint_md5(weights_path)
        with open(weights_path + ".md5", "w", encoding="utf-8") as f:
            f.write(md5 + "\n")
        print(f"  checkpoint fingerprint written: {name}.pth.md5 ({md5})")


def verify_checkpoint_md5(name: str) -> None:
    """Raise where NAME.pth does not match its fingerprint, so something overwrote it since it
    was saved on purpose. A missing sidecar also raises, and the remedy is to delete the .pth
    or re-run the crabnet_baseline step."""
    weights_path = os.path.join(_NETWORK_DIR, f"{name}.pth")
    sidecar = weights_path + ".md5"
    if not os.path.exists(sidecar):
        raise RuntimeError(
            f"CHECKPOINT UNVERIFIABLE: {sidecar} does not exist, so {name}.pth cannot "
            "be proven to be the checkpoint the baseline saved (July 2026 incident: a "
            "64-row warm-up fit auto-saved over the real weights and every transfer "
            "number was garbage). Re-run the baseline step or delete the .pth to force "
            "a fresh pretrain."
        )
    with open(sidecar, encoding="utf-8") as f:
        expected = f.read().strip()
    actual = _checkpoint_md5(weights_path)
    if actual != expected:
        raise RuntimeError(
            f"CHECKPOINT CLOBBERED: {name}.pth md5={actual} but its save-time "
            f"fingerprint is {expected}. Something overwrote the checkpoint after the "
            "baseline saved it (the July failure mode). Do not trust any result that "
            "loaded this file; re-run the baseline step."
        )
    print(f"  checkpoint verified against fingerprint: {name}.pth ({actual})")


def _apply_torch_compat_shim() -> None:
    """Empty class-level hook registries, as crabnet 2.0.8's custom optimisers subclass
    torch.optim.Optimizer without initialising them.

    Under torch 2.x every CrabNet fit after the first in one process crashes without this.
    Single-fit smoke tests pass without it, so it should not be removed on that basis.
    """
    from collections import OrderedDict
    try:
        import crabnet.utils.optim as _copt
        import crabnet.utils.utils as _cutils
    except ImportError:
        return
    for mod in (_copt, _cutils):
        for name in ("SWA", "Lamb", "Lookahead"):
            cls = getattr(mod, name, None)
            if cls is None:
                continue
            for attr in ("_optimizer_step_pre_hooks", "_optimizer_step_post_hooks"):
                if not hasattr(cls, attr):
                    setattr(cls, attr, OrderedDict())


def _load_crabnet():
    """Import the sklearn-style CrabNet class, or None with a printed hint if absent, applying
    the torch-2.x optimiser shim as a side effect."""
    try:
        from crabnet.crabnet_ import CrabNet  # type: ignore[import-untyped]
        _apply_torch_compat_shim()
        return CrabNet
    except ImportError:
        print(
            "crabnet not installed (or incompatible API). Install with `uv add crabnet` "
            "(re-run patch_elmd.py afterwards -- any uv sync rebuilds the venv), or vendor "
            "the reference implementation under external_libraries/."
        )
        return None


def _harvest_curves(cb, fold_label: str, out_dir: str) -> str | None:
    """Save any per-epoch curve attributes the fitted CrabNet holds, scanned defensively as
    they vary by version, and return the path written or None. The curve is MAE in eV rather
    than the training loss and is sampled every `checkin` epochs, so `step` shares no x-axis
    with FlexNet's per-epoch history."""
    curves = {}

    def _add(key, series):
        if isinstance(series, (list, np.ndarray)) and 1 < len(series) < 10000:
            try:
                curves[key] = np.asarray(series, dtype="float64").ravel()
            except (TypeError, ValueError):
                pass

    for name, val in vars(cb).items():
        if not any(k in name.lower() for k in ("loss", "curve", "mae", "score")):
            continue
        if isinstance(val, dict):
            for key, series in val.items():
                _add(f"{name}_{key}", series)
        else:
            _add(name, val)
    if not curves:
        return None
    df = pd.DataFrame({k: pd.Series(v) for k, v in curves.items()})
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"curves_{fold_label}.csv")
    df.to_csv(path, index_label="step")
    print(f"    curves harvested -> {path} ({list(curves)})")
    return path


def _cost(cb, train_seconds: float, n_fit_rows: int) -> dict:
    """Training-cost record for a fitted CrabNet, in the same fields the FlexNet and XGBoost
    arms write. epochs_run is the epoch the fit reached rather than the configured budget, as
    SWA aborts after three discarded updates, and attribute names differ across forks so each
    is probed and omitted rather than guessed."""
    out: dict = {"train_seconds": train_seconds, "n_fit_rows": int(n_fit_rows)}
    model = getattr(cb, "model", None)
    if model is not None and hasattr(model, "parameters"):
        try:
            out["n_params"] = int(sum(p.numel() for p in model.parameters()))
        except (TypeError, AttributeError):
            pass
    for attr, key in (("epoch", "epochs_run"), ("epochs", "epochs_budget")):
        value = getattr(cb, attr, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = int(value)
    # The sampled val curve's length shows how far the fit got where no epoch counter is exposed.
    curve = getattr(cb, "loss_curve", None)
    if isinstance(curve, dict) and isinstance(curve.get("val"), (list, np.ndarray)):
        out["n_checkins"] = int(len(curve["val"]))
    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE/RMSE/R² over all rows plus nonmetals-only variants, mirroring model.benchmark._train_fold."""
    out = {
        "mae":  float(np.abs(y_pred - y_true).mean()),
        "rmse": float(np.sqrt(((y_pred - y_true) ** 2).mean())),
        "r2":   float(r2_score(y_true, y_pred)),
    }
    mask = y_true != 0
    if mask.sum() >= 2:
        out["mae_nonmetals"]  = float(np.abs(y_pred[mask] - y_true[mask]).mean())
        out["rmse_nonmetals"] = float(np.sqrt(((y_pred[mask] - y_true[mask]) ** 2).mean()))
        out["r2_nonmetals"]   = float(r2_score(y_true[mask], y_pred[mask]))
    else:
        out["mae_nonmetals"] = out["rmse_nonmetals"] = out["r2_nonmetals"] = float("nan")
    return out


def run_crabnet_baseline(
    data_path: str = _DEFAULT_FILTERED_PATH,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    nonmetals_only: bool = False,
    seeds: tuple[int, ...] = (42, 43, 44),
    epochs: int | None = None,
    force_cpu: bool = False,
) -> pd.DataFrame | None:
    """Train CrabNet at published defaults on the exported FlexNet splits, once per seed,
    caching per-seed test metrics and predictions under cache/crabnet_results/.

    epochs=None uses the library default. seeds vary the initialisation and shuffle alone
    whilst the split stays fixed at random_state, matching how the FlexNet headline is
    reported. Returns None where crabnet is not installed.
    """
    CrabNet = _load_crabnet()
    if CrabNet is None:
        return None

    settings = {
        "val_size": val_size, "test_size": test_size, "random_state": random_state,
        "nonmetals_only": nonmetals_only, "seeds": list(seeds), "epochs": epochs,
    }
    cache_path = get_cache_path(
        "crabnet_results", [data_path], settings, extra_files=(_CRABNET_BASELINE_PY,), ext="csv"
    )
    if os.path.exists(cache_path):
        print(f"Cache valid — loading CrabNet baseline results from {cache_path}")
        return pd.read_csv(cache_path)

    split = export_splits(
        data_path=data_path, val_size=val_size, test_size=test_size,
        random_state=random_state, nonmetals_only=nonmetals_only,
    )
    train_df = pd.read_csv(split["train"])[["formula", "target"]]
    val_df   = pd.read_csv(split["val"])[["formula", "target"]]
    test_df  = pd.read_csv(split["test"])[["formula", "target"]]
    print(f"CrabNet baseline on {data_path}: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    rows: list[dict] = []
    for seed in seeds:
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)

        kwargs: dict = {"mat_prop": "band_gap", "losscurve": False, "learningcurve": False,
                        "force_cpu": force_cpu, "verbose": True, "random_state": seed,
                        "model_name": _network_name(data_path, seed)}
        if epochs is not None:
            kwargs["epochs"] = epochs
        cb = CrabNet(**kwargs)
        t0 = time.perf_counter()
        cb.fit(train_df, val_df)
        train_seconds = time.perf_counter() - t0

        pred, sigma = cb.predict(test_df, return_uncertainty=True)
        pred = np.asarray(pred, dtype="float64")
        y_true = test_df["target"].to_numpy(dtype="float64")
        metrics = _metrics(y_true, pred)
        print(f"  seed={seed}  MAE={metrics['mae']:.4f} eV  R²={metrics['r2']:.4f}  "
              f"(nonmetals-only MAE={metrics['mae_nonmetals']:.4f})  [{train_seconds:.1f}s]")
        rows.append({"seed": seed, "n_train": len(train_df), "n_test": len(test_df), **metrics,
                     **_cost(cb, train_seconds, len(train_df))})

        # Predictions and network persisted for the attention analysis and for weight reuse.
        # makedirs before the per-seed write, else the first seed crashes on a fresh machine.
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        pred_path = cache_path.replace(".csv", f"_preds_seed{seed}.csv")
        pd.DataFrame({"formula": test_df["formula"], "y_true": y_true, "y_pred": pred,
                      "sigma": np.asarray(sigma, dtype="float64")}).to_csv(
            pred_path, index=False
        )
        # Curve into the cache (2026-08-14), as crabnet's own copy is CWD-relative and overwritable.
        # See _harvest_curves for the caveat that the curve is MAE rather than loss.
        _harvest_curves(cb, f"{data_stem(data_path)}_seed{seed}", os.path.dirname(cache_path))
        try:
            # save_network() takes no argument, its explicit-name branch never defining the path.
            # The name comes from the model_name constructor kwarg.
            cb.save_network()
            write_checkpoint_md5(_network_name(data_path, seed))
        except Exception as exc:  # API without save_network, predictions still saved
            print(f"  (network not saved: {exc})")

    results = pd.DataFrame(rows)
    print("\n=== CrabNet baseline — mean across seeds ===")
    print(results[["mae", "rmse", "r2", "mae_nonmetals"]].mean().to_string())

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    results.to_csv(cache_path, index=False)
    write_meta(cache_path, [data_path], settings, extra_files=(_CRABNET_BASELINE_PY,))
    print(f"Results cached to {cache_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=_DEFAULT_FILTERED_PATH)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42, dest="random_state",
                        help="Seed fixing the train/val/test SPLIT (must match the FlexNet run)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Training seeds (weight init/shuffle) per fixed split")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override CrabNet's default epoch budget (smoke tests only)")
    parser.add_argument("--nonmetals-only", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    run_crabnet_baseline(
        data_path=args.data_file, val_size=args.val_size, test_size=args.test_size,
        random_state=args.random_state, nonmetals_only=args.nonmetals_only,
        seeds=tuple(args.seeds), epochs=args.epochs, force_cpu=args.force_cpu,
    )
