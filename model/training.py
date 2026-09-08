"""FlexNet training, covering feature preparation, the train/val/test split, the fitted
preprocessing pipeline (impute, zero-variance, correlation, scale) and the main training loop
behind train().

The pipeline is fitted on the training partition alone. The element-fraction block is exempt
from both selection steps (2026-08-17) where the caller names its columns, so held-out
chemistry reaches the model as a column seen only at zero.

model.benchmark runs a second training loop, _train_fold, kept in step with _train_one here.
"""
import itertools
import json
import os
import time
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pymatgen.core import Element
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from analysis import clean_data, filter_nonmetals, load_data, parse_compositions
from cache import get_cache_path, resolve_architecture, write_meta
from data.preprocessing import default_filtered_data_path

from config import PINNED_DATASET as _DEFAULT_FILTERED_PATH
from features.descriptors import build_nn_features
from model.network import FLEX_ACTIVATIONS, FlexNet, _CORR_THRESHOLD
from visualisation import plot_network_architecture

# Project default architecture, inherited from the removed BandGapNet's.
DEFAULT_LAYER_WIDTHS: list[int] = [256, 256, 256, 256]
DEFAULT_ACTIVATIONS: list[str] = ["snake", "snake", "relu", "relu"]

# All three stay in the cache fingerprint, else a stale .pt loads against wrong features.
_NETWORK_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network.py")
_TRAINING_PY = os.path.abspath(__file__)

# Band gaps run from 0 to roughly 20 eV, so a prediction past this bound is a runaway.
# Here rather than in model/benchmark.py, as benchmark imports from training (2026-07-28).
_DIVERGENCE_PRED_BOUND_EV = 100.0
_DESCRIPTORS_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "features", "descriptors.py"
)

_ARRAY_HISTORY_KEYS = ("val_true", "val_pred", "test_true", "test_pred")


def _model_cache_paths(
    data_path: str,
    layer_widths: list[int],
    activations: list[str],
    epochs: int,
    lr: float,
    batch_size: int,
    val_size: float,
    test_size: float,
    corr_threshold: float,
    tune: bool,
    random_state: int,
    include_elements: bool = True,
    nonmetals_only: bool = False,
    log_transform: bool = False,
    optimiser_name: str = "adam",
    negative_gap_penalty_weight: float = 0.0,
    val_checkpoint: bool = False,
) -> tuple[str, str, dict, tuple[str, ...]]:
    """Content-addressed cache locations for a trained model and its history, as (model_path,
    history_path, settings, extra_files). Every keyword argument is in the fingerprint, as each
    changes the fitted weights for the same data_path, and the defaults keep pre-existing cache
    entries valid."""
    settings = {
        "layer_widths": layer_widths, "activations": activations, "epochs": epochs, "lr": lr,
        "batch_size": batch_size, "val_size": val_size, "test_size": test_size,
        "corr_threshold": corr_threshold, "tune": tune, "random_state": random_state,
        "include_elements": include_elements,
        "nonmetals_only": nonmetals_only, "log_transform": log_transform,
        "optimiser_name": optimiser_name,
        "negative_gap_penalty_weight": negative_gap_penalty_weight,
        "val_checkpoint": val_checkpoint,
    }
    extra_files = (_NETWORK_PY, _TRAINING_PY, _DESCRIPTORS_PY)
    model_path = get_cache_path("band_gap_model", [data_path], settings, extra_files=extra_files, ext="pt")
    history_path = get_cache_path("band_gap_history", [data_path], settings, extra_files=extra_files, ext="json")
    return model_path, history_path, settings, extra_files


def _save_history(history: dict, path: str) -> None:
    """JSON-serialise a training history dict, converting numpy arrays to lists."""
    serialisable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in history.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(serialisable, f)


def write_history_preds(history: dict, history_path: str) -> str | None:
    """Flatten a finished run's per-row test predictions into <history>_preds.csv, and return
    the path written or None for a history predating the test-row keys. XGBoost and CrabNet
    write flat per-row files, so the same shape here reduces a three-model join to a read_csv."""
    needed = ("test_true", "test_pred")
    if not all(k in history for k in needed):
        return None
    n = len(history["test_true"])
    cols: dict = {}
    # material_id and formula first, so the file identifies its rows before scoring them.
    # Both are optional, as residual histories carry formulas but no material_ids.
    if len(history.get("test_material_ids") or []) == n:
        cols["material_id"] = history["test_material_ids"]
    if len(history.get("test_formulas") or []) == n:
        cols["formula"] = history["test_formulas"]
    cols["y_true"] = np.asarray(history["test_true"], dtype="float64")
    cols["y_pred"] = np.asarray(history["test_pred"], dtype="float64")
    preds_path = history_path.replace(".json", "_preds.csv")
    os.makedirs(os.path.dirname(preds_path), exist_ok=True)
    pd.DataFrame(cols).to_csv(preds_path, index=False)
    return preds_path


def _load_history(path: str) -> dict:
    """Load a training history dict, restoring numpy arrays for prediction keys."""
    with open(path) as f:
        history = json.load(f)
    for key in _ARRAY_HISTORY_KEYS:
        if key in history:
            history[key] = np.array(history[key], dtype="float32")
    return history


def prepare_features(
    data_path: str, nonmetals_only: bool = False, feature_tier: str = "standard",
    derive_vocabulary: bool | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load, clean, parse compositions and build NN features, as (X_df, y_s, formulas_s,
    material_ids_s). nonmetals_only drops Eg=0 metals first and writes its own
    content-addressed CSV, as build_nn_features caches by data-file content and keying the
    subset on the full file would collide with the full dataset's matrix."""
    df = load_data(data_path)
    valid = clean_data(df)
    feature_key_path = data_path
    if nonmetals_only:
        valid = filter_nonmetals(valid)
        variant_settings = {"nonmetals_only": True}
        feature_key_path = get_cache_path(
            "nonmetals_variant", [data_path], variant_settings, extra_files=(_TRAINING_PY,)
        )
        if not os.path.exists(feature_key_path):
            os.makedirs(os.path.dirname(feature_key_path), exist_ok=True)
            valid.to_csv(feature_key_path, index=True)
            write_meta(feature_key_path, [data_path], variant_settings, extra_files=(_TRAINING_PY,))
    compositions = parse_compositions(valid)
    X = build_nn_features(compositions, feature_key_path, feature_tier=feature_tier,
                          derive_vocabulary=derive_vocabulary)
    y = valid["band_gap"].reset_index(drop=True)
    formulas = valid["formula_pretty"].reset_index(drop=True)
    material_ids = valid.index.to_series().reset_index(drop=True)
    return X, y, formulas, material_ids


def _remove_correlated_features(X_train: np.ndarray, y_train: np.ndarray, threshold: float,
                                protected: np.ndarray | None = None) -> np.ndarray:
    """Column indices to keep, dropping one of any pair with |r| > threshold and keeping
    whichever correlates more strongly with y_train.

    Both correlation matrices use X_train and y_train alone, as this step is supervised, so
    the train-only discipline matters here in a way it does not for the steps around it.

    protected is a mask of columns that survive regardless. A column constant in train has an
    undefined correlation, where `abs(NaN) <= threshold` is False, so without the guard the
    loop would read it as correlated and undo the protection.
    """
    # errstate, as a constant protected column divides by zero here and the NaN is handled below.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(X_train, rowvar=False)
        n = corr.shape[0]
        target_corr = np.abs(np.array([np.corrcoef(X_train[:, k], y_train)[0, 1] for k in range(n)]))
    # NaN means one of the pair is constant, the opposite of redundant, so it maps to 0.
    corr = np.nan_to_num(corr, nan=0.0)
    target_corr = np.nan_to_num(target_corr, nan=0.0)
    keep_always = np.zeros(n, dtype=bool) if protected is None else protected

    to_drop: set[int] = set()
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop or abs(corr[i, j]) <= threshold:
                continue
            if keep_always[j]:
                if keep_always[i]:
                    continue  # two protected columns both stay, correlated or not
                to_drop.add(i)
                break  # i is dropped, so comparison stops and j becomes the anchor when the outer loop reaches it
            if not keep_always[i] and target_corr[j] > target_corr[i]:
                to_drop.add(i)
                break
            to_drop.add(j)
    return np.array([i for i in range(n) if i not in to_drop])


# Element symbols, used to recognise the fraction block by column name.
# No other column is a bare one- or two-letter symbol, physics columns being prefixed phys_.
_ELEMENT_SYMBOLS: frozenset[str] = frozenset(Element.from_Z(z).symbol for z in range(1, 119))


def _element_fraction_mask(feature_names: list[str] | None, n_columns: int) -> np.ndarray:
    """Boolean mask over raw columns marking the element-fraction block.

    All-False where feature_names is None, which leaves the exemption off. model.benchmark's
    _run_split and model.mdhit_arms' FlexNet arm pass no names, so the OOD suite and that arm
    run without it.
    """
    mask = np.zeros(n_columns, dtype=bool)
    if feature_names is None:
        return mask
    for i, name in enumerate(feature_names[:n_columns]):
        if name in _ELEMENT_SYMBOLS:
            mask[i] = True
    return mask


class _PreprocessingPipeline(NamedTuple):
    """Fitted impute, zero-variance, correlation and scale transform, reusable through
    .transform() on any matrix with the same raw columns. var_keep replaced a fitted
    VarianceThreshold on 2026-08-17, as the selection is now the union of nonzero-variance
    columns with the element-fraction block, so pipelines pickled before that date do not load
    into this shape and have to be rebuilt."""
    imputer: SimpleImputer
    var_keep: np.ndarray
    keep: np.ndarray
    scaler: StandardScaler
    n_after_variance: int
    feature_names: list[str] | None

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self.imputer.transform(X)
        X = X[:, self.var_keep]
        X = X[:, self.keep]
        return self.scaler.transform(X)


def _fit_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    corr_threshold: float = _CORR_THRESHOLD,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, _PreprocessingPipeline]:
    """Fit impute, zero-variance, correlation and scale on X_train alone, returning the
    transformed X_train and the fitted pipeline.

    The element-fraction block is exempt from both selection steps (2026-08-17) where
    feature_names is given, else a train-side filter makes the feature space a property of the
    split rather than of the data.
    """
    # 2. Impute, fitted on train (_fit_transform_pipeline applies it to the other splits)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)

    protected_raw = _element_fraction_mask(feature_names, X_train.shape[1])

    # 3. Remove zero-variance features, fitted on train, fractions exempt
    variances = X_train.var(axis=0)
    var_support = (variances > 0.0) | protected_raw
    var_keep = np.where(var_support)[0]
    n_protected_kept = int((protected_raw & (variances <= 0.0)).sum())
    X_train = X_train[:, var_keep]
    protected = protected_raw[var_keep]
    n_after_variance = X_train.shape[1]
    if feature_names is not None:
        feature_names = [feature_names[i] for i in var_keep]

    # 4. Remove highly correlated features, computed on train only, fractions exempt
    keep = _remove_correlated_features(X_train, y_train, corr_threshold, protected=protected)
    X_train = X_train[:, keep]
    if feature_names is not None:
        feature_names = [feature_names[i] for i in keep]

    # 5. Standardise, fitted on train.
    # StandardScaler leaves scale_=1 for a constant column, so test values are not rescaled.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    if n_protected_kept:
        print(f"  {n_protected_kept} element column(s) absent from this train partition "
              f"kept (fraction block is exempt from feature selection)")

    pipeline = _PreprocessingPipeline(
        imputer=imputer, var_keep=var_keep, keep=keep, scaler=scaler,
        n_after_variance=n_after_variance, feature_names=feature_names,
    )
    return X_train, pipeline


def _fit_transform_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *X_rest: np.ndarray,
    corr_threshold: float = _CORR_THRESHOLD,
    feature_names: list[str] | None = None,
) -> tuple[tuple[np.ndarray, ...], _PreprocessingPipeline]:
    """As _fit_pipeline, also applying the fitted transforms to each array in X_rest with
    fitting staying train-only. Returns the full fitted pipeline, as a caller that persists it
    needs the imputer and scaler state as well as the kept-column indices."""
    X_train, pipeline = _fit_pipeline(X_train, y_train, corr_threshold=corr_threshold, feature_names=feature_names)
    X_rest = tuple(pipeline.transform(X) for X in X_rest)
    return (X_train, *X_rest), pipeline


def _train_val_test_split(
    X_raw: np.ndarray,
    y_np: np.ndarray,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw train/val/test split shared by prepare_splits() and fit_dft_pipeline(), so the
    latter can reproduce the exact X_train a cached model was fit on.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test, train_idx, test_idx), the indices
    being into the original reset-index array. train_idx lets callers recover which rows a
    model was fit on, which model.residual uses for cross-stage overlap tagging.
    """
    indices = np.arange(len(y_np))
    X_tv, X_test, y_tv, y_test, idx_tv, idx_test = train_test_split(
        X_raw, y_np, indices, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val, idx_train, _ = train_test_split(
        X_tv, y_tv, idx_tv, test_size=val_size / (1.0 - test_size), random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_test


def prepare_splits(
    X_raw: np.ndarray,
    y_np: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    corr_threshold: float = _CORR_THRESHOLD,
    random_state: int = 42,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
    """Split first, then fit all preprocessing on train alone, with feature_names filtered
    alongside the data and returned as the surviving names. Returns (X_train, X_val, X_test,
    y_train, y_val, y_test, test_idx, feature_names), where test_idx indexes the original
    reset-index array."""
    X_train, X_val, X_test, y_train, y_val, y_test, _, idx_test = _train_val_test_split(
        X_raw, y_np, val_size, test_size, random_state
    )

    n_orig = X_raw.shape[1]
    (X_train, X_val, X_test), info = _fit_transform_pipeline(
        X_train, y_train, X_val, X_test, corr_threshold=corr_threshold, feature_names=feature_names
    )

    n_var  = info.n_after_variance
    n_corr = len(info.keep)
    print(
        # ASCII arrows, as cp1252 cannot encode U+2192 and the original killed the step on Windows.
        f"Feature selection: {n_orig} raw -> {n_var} (zero-variance removed) "
        f"-> {n_corr} (|r|>{corr_threshold:.2f} pairs removed)"
    )
    print(f"Split sizes — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, idx_test, info.feature_names


def fit_dft_pipeline(
    data_path: str = _DEFAULT_FILTERED_PATH,
    corr_threshold: float = _CORR_THRESHOLD,
    random_state: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
    nonmetals_only: bool = False,
) -> tuple[_PreprocessingPipeline, list[str], set[str]]:
    """Refit the pipeline a cached train() call fitted, on the same features and split, as
    (pipeline, raw_feature_names, train_compositions). It is deterministic, so the imputer,
    variance filter, correlation set and scaler are reconstructed exactly without being
    persisted. Since 2026-08-15 the fraction block is the corpus's own vocabulary, so a caller
    applying this pipeline across populations has to reindex by name first."""
    from data.dataset_analysis import _reduced_formula

    X_df, y_s, formulas_s, _ = prepare_features(data_path, nonmetals_only=nonmetals_only)
    y_np = np.asarray(y_s, dtype="float32")
    X_train, _, _, y_train, _, _, idx_train, _ = _train_val_test_split(
        X_df.values, y_np, val_size, test_size, random_state
    )
    raw_feature_names = list(X_df.columns)
    _, pipeline = _fit_pipeline(X_train, y_train, corr_threshold=corr_threshold, feature_names=raw_feature_names)
    train_formulas = formulas_s.iloc[idx_train]
    train_compositions = {rf for rf in (_reduced_formula(f) for f in train_formulas) if rf is not None}
    return pipeline, raw_feature_names, train_compositions


def _make_dataset(X: np.ndarray, y: np.ndarray, device: torch.device) -> TensorDataset:
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.float32).to(device),
    )


# SGD uses momentum=0.9, so the comparison is not against a weakened baseline.
# AdamW's weight_decay is pinned, so "adamw" tests decoupled decay rather than a default.
_OPTIMISER_FACTORIES: dict[str, Callable[..., torch.optim.Optimizer]] = {
    "adam":    lambda params, lr: torch.optim.Adam(params, lr=lr),
    "adamw":   lambda params, lr: torch.optim.AdamW(params, lr=lr, weight_decay=0.01),
    "sgd":     lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
    "rmsprop": lambda params, lr: torch.optim.RMSprop(params, lr=lr),
}
OPTIMISER_CHOICES: tuple[str, ...] = tuple(_OPTIMISER_FACTORIES)


def _make_optimiser(name: str, params, lr: float) -> torch.optim.Optimizer:
    """Shared optimiser construction, so every training loop gains a new choice together."""
    if name not in _OPTIMISER_FACTORIES:
        raise ValueError(f"Unknown optimiser_name {name!r}; choices: {OPTIMISER_CHOICES}")
    return _OPTIMISER_FACTORIES[name](params, lr)


def _train_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    layer_widths: list[int],
    activations: list[str],
    epochs: int,
    lr: float,
    batch_size: int,
    verbose: bool = True,
    epoch_callback: "Callable[[int, float], None] | None" = None,
    log_transform: bool = False,
    optimiser_name: str = "adam",
    negative_gap_penalty_weight: float = 0.0,
    val_checkpoint: bool = False,
) -> tuple[FlexNet, dict]:
    """Train one FlexNet configuration and return (model, history).

    The main training loop, used by train(), the Optuna searches, model.residual and
    model.shap_analysis. model.benchmark reimplements it in its own _train_fold.

    val_checkpoint keeps the best-val-MAE epoch's weights rather than the last epoch's, and
    log_transform fits log1p(y) whilst the history returns eV.
    negative_gap_penalty_weight is off by default, as this helper also fits signed residuals.
    """
    model = FlexNet(X_train.shape[1], layer_widths, activations, log_transform=log_transform).to(device)
    optimiser = _make_optimiser(optimiser_name, model.parameters(), lr)
    criterion = nn.MSELoss()
    y_train_fit = np.log1p(y_train) if log_transform else y_train
    y_val_fit   = np.log1p(y_val) if log_transform else y_val
    loader = DataLoader(_make_dataset(X_train, y_train_fit, device), batch_size=batch_size, shuffle=True)
    train_ds = _make_dataset(X_train, y_train_fit, device)
    val_ds = _make_dataset(X_val, y_val_fit, device)

    history: dict = {
        "epoch": [], "train_loss": [],
        "train_mae": [], "train_rmse": [], "train_r2": [],
        "val_mae": [], "val_rmse": [], "val_r2": [],
    }

    p_np: np.ndarray = np.empty(0)
    y_np: np.ndarray = np.empty(0)
    p_ev: np.ndarray = np.empty(0)
    y_ev: np.ndarray = np.empty(0)

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for Xb, yb in loader:
            optimiser.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            if negative_gap_penalty_weight > 0.0:
                loss = loss + negative_gap_penalty_weight * torch.relu(-pred).pow(2).mean()
            loss.backward()
            # Caps each step's update, so one unlucky batch cannot diverge weights to inf
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            Xv, yv = val_ds.tensors
            p_np = model(Xv).cpu().numpy()
            y_np = yv.cpu().numpy()
            # Full eval-mode pass, so train_rmse is comparable to val and test unlike train_loss.
            Xtr, ytr = train_ds.tensors
            ptr_np = model(Xtr).cpu().numpy()
            ytr_np = ytr.cpu().numpy()

        # Invert to eV immediately, so the val metrics mean the same thing either way.
        # expm1 in float64, as a diverged model overflows float32's log1p bound of roughly 88.7.
        # Deliberately no ceiling clamp, else r2_score would not raise on a diverged trial.
        if log_transform:
            p_ev = np.expm1(p_np.astype("float64"))
            y_ev = np.expm1(y_np.astype("float64"))
            ptr_ev = np.expm1(ptr_np.astype("float64"))
            ytr_ev = np.expm1(ytr_np.astype("float64"))
        else:
            p_ev = p_np
            y_ev = y_np
            ptr_ev = ptr_np
            ytr_ev = ytr_np

        train_loss = float(np.mean(batch_losses))
        mae  = float(np.abs(p_ev - y_ev).mean())
        rmse = float(np.sqrt(((p_ev - y_ev) ** 2).mean()))
        r2   = float(r2_score(y_ev, p_ev))
        train_mae  = float(np.abs(ptr_ev - ytr_ev).mean())
        train_rmse = float(np.sqrt(((ptr_ev - ytr_ev) ** 2).mean()))
        train_r2   = float(r2_score(ytr_ev, ptr_ev))

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_mae"].append(train_mae)
        history["train_rmse"].append(train_rmse)
        history["train_r2"].append(train_r2)
        history["val_mae"].append(mae)
        history["val_rmse"].append(rmse)
        history["val_r2"].append(r2)

        if epoch_callback is not None:
            epoch_callback(epoch, mae)

        if val_checkpoint and mae <= min(history["val_mae"]):
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(
                f"  Epoch {epoch:>4}/{epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"train RMSE={train_rmse:.4f} eV  "
                f"val MAE={mae:.4f} eV  "
                f"RMSE={rmse:.4f} eV  "
                f"R²={r2:.4f}"
            )

    if val_checkpoint and "best_state" in dir():
        model.load_state_dict(best_state)
        history["checkpoint_epoch"] = best_epoch

    # Final val predictions for the parity plot, in eV per the rule above.
    history["val_true"] = y_ev
    history["val_pred"] = p_ev
    return model, history


def _tune_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    param_grid: dict,
    epochs: int,
    batch_size: int,
    log_transform: bool = False,
) -> dict:
    """Grid search over param_grid's full cross-product, scored on val MAE alone as the test set is never seen here."""
    combos = list(itertools.product(*param_grid.values()))
    keys   = list(param_grid.keys())
    print(f"Hyperparameter search: {len(combos)} configs × {epochs} epochs each")

    best_mae    = float("inf")
    best_params: dict = {}

    for combo in combos:
        params = dict(zip(keys, combo))
        _, history = _train_one(
            X_train, y_train, X_val, y_val, device,
            layer_widths=params["layer_widths"],
            activations=params["activations"],
            epochs=epochs,
            lr=params["lr"],
            batch_size=batch_size,
            verbose=False,
            log_transform=log_transform,
        )
        mae = history["val_mae"][-1]
        print(
            f"  layer_widths={params['layer_widths']}  "
            f"activations={params['activations']}  "
            f"lr={params['lr']:.0e}  "
            f"val MAE={mae:.4f} eV"
        )
        if mae < best_mae:
            best_mae    = mae
            best_params = params

    print(f"Best: {best_params}  (val MAE={best_mae:.4f} eV)")
    return best_params


def train(
    data_path: str = _DEFAULT_FILTERED_PATH,
    layer_widths: list[int] | None = None,
    activations: list[str] | None = None,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    val_size: float = 0.15,
    test_size: float = 0.15,
    corr_threshold: float = _CORR_THRESHOLD,
    tune: bool = False,
    random_state: int = 42,
    use_best_found: bool = False,
    nonmetals_only: bool = False,
    log_transform: bool = False,
    optimiser_name: str = "adam",
    negative_gap_penalty_weight: float = 0.05,
    val_checkpoint: bool = False,
    init_seed: "int | None" = None,
    feature_tier: str = "standard",
) -> tuple[FlexNet, dict]:
    """Train FlexNet, or load a cached model and history for this configuration, as
    (model, history).

    random_state seeds the split and the initialisation together, whilst init_seed decouples
    them, allowing deep ensembles over a fixed split. Both it and a non-standard feature_tier
    are in the cache key.

    log_transform fits log1p(Eg) internally and returns eV, and must not be set when training
    on signed residuals. use_best_found was deprecated on 2026-07-17 and raises.
    """
    if layer_widths is None:
        layer_widths = DEFAULT_LAYER_WIDTHS
    if activations is None:
        activations = DEFAULT_ACTIVATIONS
    layer_widths, activations, lr, optimiser_name = resolve_architecture(
        layer_widths, activations, lr, use_best_found, optimiser_name
    )

    model_path, history_path, cache_settings, extra_files = _model_cache_paths(
        data_path, layer_widths, activations, epochs, lr, batch_size,
        val_size, test_size, corr_threshold, tune, random_state,
        nonmetals_only=nonmetals_only, log_transform=log_transform, optimiser_name=optimiser_name,
        negative_gap_penalty_weight=negative_gap_penalty_weight,
        val_checkpoint=val_checkpoint,
    )
    if init_seed is not None:
        cache_settings["init_seed"] = init_seed
    if feature_tier != "standard":  # ladder tiers get their own model/history keys
        cache_settings["feature_tier"] = feature_tier
    if init_seed is not None or feature_tier != "standard":
        import hashlib as _h, json as _j
        _hash = _h.md5(_j.dumps(cache_settings, sort_keys=True, default=str).encode()).hexdigest()[:12]
        model_path = os.path.join(os.path.dirname(model_path), _hash + ".pt")
        history_path = os.path.join(os.path.dirname(history_path), _hash + ".json")

    X_df, y_s, formulas_s, material_ids_s = prepare_features(
        data_path, nonmetals_only=nonmetals_only, feature_tier=feature_tier
    )
    y_np: np.ndarray = np.asarray(y_s, dtype="float32")
    formulas_arr = np.array(formulas_s.tolist())
    material_ids_arr = np.array(material_ids_s.tolist())

    X_train, X_val, X_test, y_train, y_val, y_test, test_idx, feature_names = prepare_splits(
        X_df.values, y_np,
        val_size=val_size,
        test_size=test_size,
        corr_threshold=corr_threshold,
        random_state=random_state,
        feature_names=list(X_df.columns),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(model_path) and os.path.exists(history_path):
        print(f"Cache valid — loading trained model from {model_path}")
        model = FlexNet(X_train.shape[1], layer_widths, activations, log_transform=log_transform).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        return model, _load_history(history_path)

    print(f"Training on {device}  |  {X_train.shape[1]} input features  |  seed={random_state}")

    if tune:
        torch.manual_seed(random_state)
        param_grid = {
            "layer_widths": [[128] * 4, [256] * 4, [512] * 4],
            "activations":  [DEFAULT_ACTIVATIONS],
            "lr":           [1e-3, 3e-4],
        }
        best = _tune_hyperparams(
            X_train, y_train, X_val, y_val, device,
            param_grid=param_grid,
            epochs=max(50, epochs // 4),
            batch_size=batch_size,
            log_transform=log_transform,
        )
        layer_widths = best["layer_widths"]
        activations  = best["activations"]
        lr           = best["lr"]
        print(f"\nRetraining with best params: layer_widths={layer_widths}  activations={activations}  lr={lr:.0e}")

    # Re-seed, so initialisation depends on the seed alone rather than on the tuning search.
    torch.manual_seed(random_state if init_seed is None else init_seed)
    _t_fit_start = time.perf_counter()
    model, history = _train_one(
        X_train, y_train, X_val, y_val, device,
        layer_widths=layer_widths,
        activations=activations,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        verbose=True,
        log_transform=log_transform,
        optimiser_name=optimiser_name,
        negative_gap_penalty_weight=negative_gap_penalty_weight,
        val_checkpoint=val_checkpoint,
    )

    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        yt = torch.tensor(y_test, dtype=torch.float32).to(device)
        raw_preds  = model(Xt).cpu().numpy()  # log1p(Eg) if log_transform, else raw eV
        # float64 with no clamp, following _train_one's val and train inversion above
        test_preds = np.expm1(raw_preds.astype("float64")) if log_transform else raw_preds
        y_test_np  = yt.cpu().numpy()  # always raw eV, never transformed outside _train_one

    test_mae  = float(np.abs(test_preds - y_test_np).mean())
    test_rmse = float(np.sqrt(((test_preds - y_test_np) ** 2).mean()))
    test_r2   = float(r2_score(y_test_np, test_preds))

    # MAPE excludes zero-gap (PBE metallic) samples to avoid division by zero
    nonzero = y_test_np != 0.0
    if nonzero.sum() > 0:
        test_mape = float(
            np.mean(np.abs((test_preds[nonzero] - y_test_np[nonzero]) / np.abs(y_test_np[nonzero]))) * 100
        )
    else:
        test_mape = float("nan")

    n_zero = int((~nonzero).sum())
    print(
        f"\nTest set  MAE={test_mae:.4f} eV  "
        f"RMSE={test_rmse:.4f} eV  "
        f"R²={test_r2:.4f}  "
        f"MAPE={test_mape:.2f}%  "
        f"(n={len(y_test)}, MAPE excludes {n_zero} zero-gap metals)"
    )

    # Divergence flag, mirroring model.benchmark._train_fold's (added here 2026-07-28).
    # MAE is robust to a few exploded predictions, so a diverged run hides in a multi-seed mean.
    # Under log_transform a large output in log space becomes huge after expm1.
    # Flagged here, so a blow-up reaches the run log and the .meta sidecar.
    diverged = bool(
        not np.isfinite(test_r2)
        or test_r2 < 0.0
        or not np.isfinite(test_preds).all()
        or float(np.abs(test_preds).max()) > _DIVERGENCE_PRED_BOUND_EV
    )
    if diverged:
        print(
            f"  WARNING: run flagged DIVERGED (R2={test_r2:.4f}, "
            f"max |prediction|={float(np.abs(test_preds).max()):.1f} eV, bound "
            f"{_DIVERGENCE_PRED_BOUND_EV} eV). Do not average this seed into a reported "
            "mean without saying so."
        )

    history["test_mae"]          = test_mae
    history["test_rmse"]         = test_rmse
    history["test_r2"]           = test_r2
    history["test_mape"]         = test_mape
    history["diverged"]          = diverged
    history["test_true"]         = y_test_np
    history["test_pred"]         = test_preds
    history["test_formulas"]     = formulas_arr[test_idx].tolist()
    history["test_material_ids"] = material_ids_arr[test_idx].tolist()
    history["n_train"]           = len(y_train)
    history["feature_names"]     = feature_names
    history["n_raw_features"]    = X_df.shape[1]
    # Training cost (2026-08-14), the same fields the benchmark folds and the other arms record.
    # train_seconds is the _train_one call alone, as features and preprocessing are cached.
    # best_epoch equals epochs unless val_checkpoint is on.
    history["train_seconds"]     = time.perf_counter() - _t_fit_start
    history["n_params"]          = int(sum(p.numel() for p in model.parameters()))
    history["epochs_run"]        = len(history["epoch"])
    history["best_epoch"]        = (
        int(np.argmin(history["val_mae"]) + 1) if val_checkpoint and history.get("val_mae")
        else len(history["epoch"])
    )
    history["device"]            = str(device)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    _save_history(history, history_path)
    write_meta(model_path, [data_path], cache_settings, extra_files=extra_files)
    write_meta(history_path, [data_path], cache_settings, extra_files=extra_files)
    preds_path = write_history_preds(history, history_path)
    if preds_path:
        print(f"Per-material test predictions cached to {preds_path} ({len(y_test)} rows)")
    print(f"Model cached to {model_path}")

    return model, history


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file",       default=_DEFAULT_FILTERED_PATH)
    parser.add_argument("--layer-widths",    type=int, nargs="+", default=DEFAULT_LAYER_WIDTHS,
                         help="One width per hidden layer")
    parser.add_argument("--activations",     nargs="+", choices=list(FLEX_ACTIVATIONS), default=DEFAULT_ACTIVATIONS,
                         help="One activation per hidden layer, same length as --layer-widths")
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch-size",      type=int,   default=256)
    parser.add_argument("--val-size",        type=float, default=0.15)
    parser.add_argument("--test-size",       type=float, default=0.15)
    parser.add_argument("--corr-threshold",  type=float, default=_CORR_THRESHOLD)
    parser.add_argument("--tune",            action="store_true")
    parser.add_argument("--seed",            type=int,   default=42, dest="random_state")
    parser.add_argument("--nonmetals-only",  action="store_true",
                         help="Drop Eg=0 metals before training (BGML-style nonmetal-only regression)")
    parser.add_argument("--log-transform",   action="store_true",
                         help="Fit the network against log1p(Eg) instead of raw eV, to weight "
                              "errors evenly across Eg's long right tail")
    parser.add_argument("--show",            action="store_true", dest="show_figures")
    args = parser.parse_args()

    if len(args.activations) != len(args.layer_widths):
        parser.error(
            f"--activations must be given with one entry per --layer-widths entry "
            f"({len(args.layer_widths)}), got {args.activations!r}"
        )

    model, history = train(
        data_path=args.data_file,
        layer_widths=args.layer_widths,
        activations=args.activations,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_size=args.val_size,
        test_size=args.test_size,
        corr_threshold=args.corr_threshold,
        tune=args.tune,
        random_state=args.random_state,
        nonmetals_only=args.nonmetals_only,
        log_transform=args.log_transform,
    )

    from visualisation import plot_learning_curves, plot_parity, plot_error_distribution, plot_parity_by_gap_magnitude
    plot_network_architecture(model)
    plot_learning_curves(history)
    plot_parity(history["test_true"], history["test_pred"])
    plot_error_distribution(history["test_true"], history["test_pred"])
    plot_parity_by_gap_magnitude(history["test_true"], history["test_pred"])

    if args.show_figures:
        plt.show()
