"""OOD benchmark for FlexNet, following Wang et al. (2026) "Benchmarking bandgap
prediction in semiconductors under experimental and realistic evaluation settings", with
BENCHMARK_FIDELITY.md recording the deviations.

Six split strategies are evaluated (random, lomo, chemical_system and periodic_group through
MatFold, crystal_system through GroupKFold, and a data-efficiency sweep over 10%, 25%, 50% and
100% of the random split's training data). feature_ood is excluded by design, with the reason
in model/folds.py's EXCLUDED_SPLITS.

Folds come from the shared artifact in model/folds.py, which all three model runners read.
"""

import os
import re
import time

import joblib
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from MatFold import MatFold
from pymatgen.core import Composition, Lattice, Structure
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from analysis import clean_data, filter_nonmetals, load_data, parse_compositions
from cache import CACHE_DIR, get_cache_path, resolve_architecture, write_meta
from config import RAW_DATA_PATH
from data.preprocessing import default_filtered_data_path

from config import PINNED_DATASET as _DEFAULT_FILTERED_PATH
from features.descriptors import build_nn_features
from model.network import FLEX_ACTIVATIONS, FlexNet, _CORR_THRESHOLD
from model.training import (
    DEFAULT_ACTIVATIONS,
    DEFAULT_LAYER_WIDTHS,
    OPTIMISER_CHOICES,
    _fit_transform_pipeline,
    _make_dataset,
    _make_optimiser,
    prepare_features,
    train as train_model,
)

_CATEGORY_ELEMENTS: dict[str, set[str]] = {
    "Chalcogenides": {"S", "Se", "Te"},
    "Oxides":        {"O"},
    "Halides":       {"F", "Cl", "Br", "I"},
    "Nitrides":      {"N"},
    "Phosphides":    {"P"},
    "Arsenides":     {"As"},
    "Antimonides":   {"Sb"},
    "Silicides":     {"Si"},
    "Carbides":      {"C"},
    "Hydrides":      {"H"},
}

_NETWORK_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network.py")
_BENCHMARK_PY = os.path.abspath(__file__)
# training.py stays in the key, else the outer CSV cache would never re-ask an invalidated train().
_TRAINING_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training.py")
# folds.py in run_benchmark's key (2026-07-27), else a stale CSV masks a changed split recipe.
_FOLDS_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "folds.py")
_MATFOLD_TMP_DIR = os.path.join(CACHE_DIR, "matfold_tmp")
# Moved to model/training.py 2026-07-28, so train() and _train_fold share one bound.
# Re-exported here, as this module's own code and callers already reference it.
from model.training import _DIVERGENCE_PRED_BOUND_EV  # noqa: E402


def mrae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Relative Absolute Error, Wang et al. (2026)'s primary metric, excluding zero-gap metallic samples to avoid division by zero."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / np.abs(y_true[mask])))


def _bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, random_state: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of values, so two splits can be compared by CI
    overlap rather than by two point estimates, and (nan, nan) where there are fewer than two
    values."""
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_state)
    boot_means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def classify_material(formula: str) -> str:
    """Map a formula to a material category, following Wang et al.'s scheme, giving 'multi' where several match (excluded from LOMO) and 'Others' where none do."""
    elements = {el.symbol for el in Composition(formula).elements}
    matching = [cat for cat, keys in _CATEGORY_ELEMENTS.items() if elements & keys]
    if len(matching) == 0:
        return "Others"
    if len(matching) == 1:
        return matching[0]
    return "multi"


def _preprocess_fold(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    corr_threshold: float = _CORR_THRESHOLD,
    return_pipeline: bool = False,
    feature_names: list[str] | None = None,
):
    """Preprocess one train/test fold through _fit_transform_pipeline, fitting on train alone,
    where y_raw feeds the correlation step's tie-break and nothing else. return_pipeline also
    returns the fitted pipeline, needed to apply a saved fold model to new compositions."""
    X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
    y_tr = y_raw[train_idx]
    (X_tr, X_te), pipeline = _fit_transform_pipeline(
        X_tr, y_tr, X_te, corr_threshold=corr_threshold, feature_names=feature_names
    )
    return (X_tr, X_te, pipeline) if return_pipeline else (X_tr, X_te)


def random_stratified_split(
    indices: np.ndarray,
    categories: np.ndarray,
    test_size: float = 0.1,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random stratified split by material category, as a single train/test fold."""
    strat = np.where(np.isin(categories, ["multi", "Others"]), "Others", categories)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=strat
    )
    return [(train_idx, test_idx)]


def data_efficiency_splits(
    indices: np.ndarray,
    categories: np.ndarray,
    fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
    test_size: float = 0.1,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Data-efficiency sweep following Wang et al., on the same fixed test set as
    random_stratified_split with the training side subsampled to each fraction, returning
    (train_idx, test_idx, fraction_label) folds."""
    strat = np.where(np.isin(categories, ["multi", "Others"]), "Others", categories)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=strat
    )
    train_strat = strat[train_idx]
    folds: list[tuple[np.ndarray, np.ndarray, str]] = []
    for frac in fractions:
        if frac >= 1.0:
            sub_idx = train_idx
        else:
            sub_idx, _ = train_test_split(
                train_idx, train_size=frac, random_state=random_state, stratify=train_strat
            )
        folds.append((sub_idx, test_idx, f"{int(round(frac * 100))}pct"))
    return folds


def feature_ood_split(
    X_imputed: np.ndarray,
    indices: np.ndarray,
    k: int = 8,
    random_state: int = 42,
    min_test_size: int = 30,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Feature OOD split by k-means, as one train/test fold or none where the farthest
    cluster holds fewer than min_test_size points.

    No longer reached, as the split was excluded by design on 2026-07-27, and kept so the
    campaigns that did run it stay readable (model/folds.py's EXCLUDED_SPLITS).

    The guard was added 2026-07-23, as that cluster can be a lone outlier, and returning []
    leaves the absent rows visible.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    global_mean = X_scaled.mean(axis=0)
    distances = np.linalg.norm(kmeans.cluster_centers_ - global_mean, axis=1)
    ood_cluster = int(np.argmax(distances))

    ood_mask = labels == ood_cluster
    n_test = int(ood_mask.sum())
    if n_test < min_test_size:
        print(
            f"\n!!! FEATURE-OOD SPLIT SKIPPED: farthest k-means cluster (k={k}, "
            f"random_state={random_state}) holds {n_test} sample(s) < "
            f"min_test_size={min_test_size}. A fold this small is one compound's "
            "error, not a benchmark. No feature_ood rows will be written; if this "
            "split is needed, retune k/random_state deliberately and log the change "
            "in DECISIONS.md.\n"
        )
        return []
    print(f"  Feature OOD: cluster {ood_cluster} selected ({n_test} test samples)")
    return [(indices[~ood_mask], indices[ood_mask])]


def lomo_splits(
    indices: np.ndarray,
    categories: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Leave-one-material-category-out, excluding the 'multi' and 'Others' categories."""
    folds: list[tuple[np.ndarray, np.ndarray, str]] = []
    for cat in sorted(set(categories) - {"multi", "Others"}):
        mask = categories == cat
        if mask.sum() == 0:
            continue
        folds.append((indices[~mask], indices[mask], cat))
    return folds


def _dummy_structure_dict(formula: str) -> dict:
    """Minimal placeholder Structure carrying the formula's composition alone, as MatFold's
    chemsys and periodictablegroups splits read structure.composition and never geometry. Its
    sgnum, pointgroup and crystalsys columns are meaningless here, and only
    crystal_system_splits uses real symmetry."""
    comp = Composition(formula)
    symbols = list(comp.get_el_amt_dict().keys())
    n = max(len(symbols), 1)
    lattice = Lattice.cubic(5.0 * n)
    coords = [[i / n, 0.0, 0.0] for i in range(n)]
    return Structure(lattice, symbols, coords).as_dict()


def _build_matfold(indices: np.ndarray, formulas: np.ndarray) -> MatFold:
    """Build a MatFold instance from formula strings via placeholder structures, where id_ is
    str(position) so folds read back by _matfold_splits are usable as index arrays."""
    struct_cache: dict[str, dict] = {}
    ids = [str(i) for i in range(len(indices))]
    bulk_dict: dict[str, dict] = {}
    for id_, f in zip(ids, formulas):
        if f not in struct_cache:
            struct_cache[f] = _dummy_structure_dict(f)
        bulk_dict[id_] = struct_cache[f]
    df = pd.DataFrame({"id": ids})
    # seed=42 matches Wang et al.'s stated reproducibility seed (MatFold itself defaults to 0)
    return MatFold(df, bulk_dict, write_data_checksums=False, seed=42)


def _matfold_splits(
    mf: MatFold,
    split_type: str,
    n_splits: int = 10,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run a MatFold k-fold split and parse the CSVs it writes into (train_idx, test_idx) arrays, adapting k down where fewer unique groups exist."""
    n_possible = len(mf.split_statistics(split_type))
    k = min(n_splits, n_possible)
    if k < 2:
        return []

    os.makedirs(_MATFOLD_TMP_DIR, exist_ok=True)
    write_base = f"mf_{split_type}"
    mf.create_nested_splits(
        split_type, n_inner_splits=1, n_outer_splits=k,
        output_dir=_MATFOLD_TMP_DIR, write_base_str=write_base, verbose=False,
    )
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        train_csv = os.path.join(_MATFOLD_TMP_DIR, f"{write_base}.{split_type}.k{i}_outer.train.csv")
        test_csv = os.path.join(_MATFOLD_TMP_DIR, f"{write_base}.{split_type}.k{i}_outer.test.csv")
        if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
            continue
        train_idx = pd.read_csv(train_csv)["id"].astype(int).to_numpy()
        test_idx = pd.read_csv(test_csv)["id"].astype(int).to_numpy()
        folds.append((train_idx, test_idx))
    return folds


def chemical_system_splits(mf: MatFold, n_splits: int = 10) -> list[tuple[np.ndarray, np.ndarray]]:
    """MatFold k-fold split by chemical system, the elements present ignoring stoichiometry."""
    return _matfold_splits(mf, "chemsys", n_splits)


def periodic_group_splits(mf: MatFold, n_splits: int = 10) -> list[tuple[np.ndarray, np.ndarray]]:
    """MatFold k-fold split by periodic-table group of constituent elements."""
    return _matfold_splits(mf, "periodictablegroups", n_splits)


def crystal_system_splits(
    indices: np.ndarray,
    crystal_systems: pd.Series,
    n_splits: int = 7,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """GroupKFold by crystal system, or None where there is no valid data or fewer than two distinct systems to group on."""
    valid_mask: np.ndarray = np.asarray(pd.notna(crystal_systems))
    idx = indices[valid_mask]
    grp: np.ndarray = np.asarray(crystal_systems[valid_mask].astype(str))
    if len(idx) == 0:
        return None
    k = min(n_splits, len(set(grp)))
    if k < 2:
        return None
    gkf = GroupKFold(n_splits=k)
    pos = np.arange(len(idx))
    return [(idx[tr], idx[te]) for tr, te in gkf.split(pos, groups=grp)]


def _fetch_mp_symmetry(
    material_ids: list[str],
    cache_path: str,
    *,
    include_spacegroup: bool,
    label: str,
) -> pd.DataFrame:
    """Fetch symmetry fields from the Materials Project API and cache to CSV."""
    from dotenv import load_dotenv
    from mp_api.client import MPRester

    load_dotenv()
    chunk_size = 500
    all_records: list[dict] = []
    with MPRester(os.getenv("MP_API_KEY")) as mpr:
        for i in range(0, len(material_ids), chunk_size):
            chunk = material_ids[i : i + chunk_size]
            print(f"  Fetching {label} {i + 1}–{min(i + chunk_size, len(material_ids))} / {len(material_ids)}...")
            results = mpr.materials.summary.search(
                material_ids=chunk,
                fields=["material_id", "symmetry"],
                all_fields=False,
            )
            for r in results:
                sym = r.symmetry
                record = {
                    "material_id":    r.material_id,
                    "crystal_system": sym.crystal_system.value if sym else None,
                }
                if include_spacegroup:
                    record["spacegroup_symbol"] = sym.symbol if sym else None
                all_records.append(record)
    df = pd.DataFrame(all_records).set_index("material_id")
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    df.to_csv(cache_path)
    return df


def _fetch_crystal_systems(material_ids: list[str], cache_path: str) -> pd.DataFrame:
    """Fetch crystal systems from the Materials Project API and cache to CSV."""
    return _fetch_mp_symmetry(material_ids, cache_path, include_spacegroup=False, label="crystal systems")


def _load_crystal_systems(
    df: pd.DataFrame,
    data_path: str,
) -> pd.Series | None:
    """Crystal system per material_id, from the dataframe, the cache, or the MP API."""
    cache_path = get_cache_path("crystal_systems", [data_path], {}, ext="csv")
    if "crystal_system" in df.columns:
        return df["crystal_system"]

    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col="material_id")
        return df.index.to_series().map(cached["crystal_system"].to_dict())

    print("Crystal system data not found — fetching from Materials Project API...")
    try:
        cs_df = _fetch_crystal_systems(list(df.index), cache_path)
        return df.index.to_series().map(cs_df["crystal_system"].to_dict())
    except Exception as exc:
        print(f"Warning: crystal system fetch failed ({exc}). Skipping crystal system split.")
        return None


def _train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    layer_widths: list[int],
    activations: list[str],
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    log_transform: bool = False,
    optimiser_name: str = "adam",
    random_state: int = 42,
    negative_gap_penalty_weight: float = 0.0,
    val_checkpoint: bool = False,
    val_fraction: float = 0.1,
) -> tuple[dict[str, float], np.ndarray, dict]:
    """Train one fold and return (metrics, preds, extras), where extras holds the
    per-epoch history, the model and the cost.

    A second training loop, kept in step with model.training._train_one so benchmark folds
    train under the headline recipe.

    val_checkpoint carves val_fraction off train and keeps the best epoch's weights, the rule
    XGBoost and CrabNet already had on these folds. It is fingerprinted, so a checkpointed
    suite lands beside the fixed-epoch one, and it costs training rows, so a change in fold MAE
    confounds better stopping with less data (DECISIONS.md 2026-08-14).

    metrics carries *_nonmetals variants dropping zero-gap rows, else splits differing in metal
    fraction would differ in MAE for that reason alone.
    """
    torch.manual_seed(random_state)

    # Carved out before the model is built, so the fit sees the reduced training set alone.
    # A dedicated Generator leaves the torch RNG stream alone, so initialisation is unchanged.
    X_fit, y_fit = X_train, y_train
    X_val_t = y_val_ev = None
    if val_checkpoint:
        n_val = int(round(len(y_train) * val_fraction))
        if n_val >= 2 and len(y_train) - n_val >= 2:
            order = np.random.default_rng(random_state).permutation(len(y_train))
            val_idx, fit_idx = order[:n_val], order[n_val:]
            X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
            X_val_t = torch.tensor(X_train[val_idx], dtype=torch.float32).to(device)
            y_val_ev = y_train[val_idx]
        else:
            # Tiny folds cannot spare a split, so training is fixed-epoch and best_epoch == epochs.
            print(f"    (fold too small for a {val_fraction:.0%} val split -- fixed-epoch)")

    model = FlexNet(X_fit.shape[1], layer_widths, activations, log_transform=log_transform).to(device)
    opt = _make_optimiser(optimiser_name, model.parameters(), lr)
    loss_fn = nn.MSELoss()
    y_fit_transformed = np.log1p(y_fit) if log_transform else y_fit
    loader = DataLoader(
        _make_dataset(X_fit, y_fit_transformed, device), batch_size=batch_size, shuffle=True
    )

    X_train_t = torch.tensor(X_fit, dtype=torch.float32).to(device)
    y_train_fit_t = torch.tensor(y_fit_transformed, dtype=torch.float32).to(device)
    history: dict[str, list[float]] = {"train_loss": [], "train_rmse_fit": []}
    if X_val_t is not None:
        history["val_mae"], history["val_rmse"] = [], []

    best_val, best_epoch, best_state = float("inf"), epochs, None
    n_params = int(sum(p.numel() for p in model.parameters()))
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for Xb, yb in loader:
            opt.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            if negative_gap_penalty_weight > 0.0:
                loss = loss + negative_gap_penalty_weight * torch.relu(-pred).pow(2).mean()
            loss.backward()
            # The same cap as model.training._train_one, so one batch cannot diverge weights.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += float(loss.detach()); n_batches += 1
        history["train_loss"].append(epoch_loss / max(n_batches, 1))
        model.eval()
        with torch.no_grad():
            resid = model(X_train_t) - y_train_fit_t
            history["train_rmse_fit"].append(float(torch.sqrt((resid ** 2).mean())))
            if X_val_t is not None:
                raw_val = model(X_val_t).cpu().numpy()
                # Back to eV in float64, else a diverging fold picks its best epoch on inf.
                p_val = np.expm1(raw_val.astype("float64")) if log_transform else raw_val
                val_mae = float(np.abs(p_val - y_val_ev).mean())
                history["val_mae"].append(val_mae)
                history["val_rmse"].append(float(np.sqrt(((p_val - y_val_ev) ** 2).mean())))
                # Strict <, so ties keep the earlier epoch and the cheaper model is kept.
                if val_mae < best_val:
                    best_val, best_epoch = val_mae, epoch
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - t_start

    model.eval()
    with torch.no_grad():
        raw_preds = (
            model(torch.tensor(X_test, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )
    # expm1 in float64, matching _train_one, as float32 overflows above roughly 88 in log space.
    preds = np.expm1(raw_preds.astype("float64")) if log_transform else raw_preds

    metrics = {
        "mae":  float(np.abs(preds - y_test).mean()),
        "rmse": float(np.sqrt(((preds - y_test) ** 2).mean())),
        "r2":   float(r2_score(y_test, preds)),
        "mrae": mrae(y_test, preds),
    }
    nonmetal_mask = y_test != 0
    if nonmetal_mask.sum() >= 2:
        yt_nm, yp_nm = y_test[nonmetal_mask], preds[nonmetal_mask]
        metrics["mae_nonmetals"]  = float(np.abs(yp_nm - yt_nm).mean())
        metrics["rmse_nonmetals"] = float(np.sqrt(((yp_nm - yt_nm) ** 2).mean()))
        metrics["r2_nonmetals"]   = float(r2_score(yt_nm, yp_nm))
    else:
        metrics["mae_nonmetals"] = metrics["rmse_nonmetals"] = metrics["r2_nonmetals"] = float("nan")

    # Divergence flag, as since the float64 fix a diverged fold gives a large finite error.
    # R2 < 0 is worse than predicting the mean, and the bound catches runaway outputs.
    metrics["diverged"] = bool(
        not np.isfinite(metrics["r2"])
        or metrics["r2"] < 0.0
        or not np.isfinite(preds).all()
        or float(np.abs(preds).max()) > _DIVERGENCE_PRED_BOUND_EV
    )

    # Cost record (2026-08-14), beside the metrics rather than printed, as the thesis may cite it.
    # train_seconds covers the fit alone, as features and preprocessing are shared and cached.
    cost = {
        "train_seconds": train_seconds,
        "n_params": n_params,
        "n_fit_rows": int(len(y_fit)),
        "epochs_run": epochs,
        "best_epoch": best_epoch,
        "best_val_mae": best_val if best_state is not None else float("nan"),
    }
    return metrics, preds, {"history": history, "model": model, "cost": cost}


def _run_split(
    name: str,
    folds: list,
    X_raw: np.ndarray,
    y_np: np.ndarray,
    device: torch.device,
    layer_widths: list[int],
    activations: list[str],
    epochs: int,
    lr: float,
    batch_size: int,
    labeled_folds: bool = False,
    log_transform: bool = False,
    optimiser_name: str = "adam",
    seeds: tuple[int, ...] = (42, 43, 44),
    negative_gap_penalty_weight: float = 0.0,
    val_checkpoint: bool = False,
    val_fraction: float = 0.1,
    pred_sink: list | None = None,
    history_sink: list | None = None,
    model_dir: str | None = None,
) -> list[dict]:
    """Train every fold of one split strategy, returning one unaveraged row per (fold, seed),
    with labeled_folds True where fold entries carry a label. pred_sink collects per-row
    predictions and history_sink the per-epoch curves, without which only fold-level scalars
    survive, and model_dir saves each fold's weights and fitted pipeline."""
    rows: list[dict] = []
    for fold_idx, fold in enumerate(folds):
        if labeled_folds:
            train_idx, test_idx, fold_label = fold
        else:
            train_idx, test_idx = fold
            fold_label = str(fold_idx)

        X_tr, X_te, fold_pipeline = _preprocess_fold(
            X_raw, y_np, train_idx, test_idx, return_pipeline=True
        )
        y_tr, y_te = y_np[train_idx], y_np[test_idx]

        print(f"  [{name}] fold={fold_label}  train={len(y_tr)}  test={len(y_te)}")
        fold_maes: list[float] = []
        for seed in seeds:
            metrics, preds, extras = _train_fold(
                X_tr, y_tr, X_te, y_te, device,
                layer_widths=layer_widths, activations=activations,
                epochs=epochs, lr=lr, batch_size=batch_size,
                log_transform=log_transform,
                optimiser_name=optimiser_name,
                random_state=seed,
                negative_gap_penalty_weight=negative_gap_penalty_weight,
                val_checkpoint=val_checkpoint,
                val_fraction=val_fraction,
            )
            if pred_sink is not None:
                pred_sink.append(pd.DataFrame({
                    "split": name, "fold": fold_label, "seed": seed,
                    "row_idx": np.asarray(test_idx), "y_true": y_te, "y_pred": preds,
                }))
            if history_sink is not None:
                h = extras["history"]
                frame = {
                    "split": name, "fold": fold_label, "seed": seed,
                    "epoch": np.arange(1, len(h["train_loss"]) + 1),
                    "train_loss": h["train_loss"], "train_rmse_fit": h["train_rmse_fit"],
                }
                # Present under val_checkpoint alone, so the file itself says which arm produced it.
                for key in ("val_mae", "val_rmse"):
                    if key in h:
                        frame[key] = h[key]
                history_sink.append(pd.DataFrame(frame))
            if model_dir is not None:
                safe = re.sub(r"[^A-Za-z0-9._-]", "_", f"{name}_{fold_label}_seed{seed}")
                os.makedirs(model_dir, exist_ok=True)
                torch.save(extras["model"].state_dict(), os.path.join(model_dir, f"{safe}.pt"))
                joblib.dump(fold_pipeline, os.path.join(model_dir, f"{safe}_pipeline.joblib"))
            print(
                f"    seed={seed}  MAE={metrics['mae']:.4f}  MRAE={metrics['mrae']:.4f}  "
                f"R²={metrics['r2']:.4f}  (nonmetals-only MAE={metrics['mae_nonmetals']:.4f})"
            )
            fold_maes.append(metrics["mae"])
            rows.append({
                "split": name, "fold": fold_label, "seed": seed,
                "n_train": len(y_tr), "n_test": len(y_te),
                **metrics,
                **extras["cost"],
            })
        if len(seeds) > 1:
            print(f"    -> {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f} MAE across {len(seeds)} seeds")
    return rows


def run_benchmark(
    data_path: str = _DEFAULT_FILTERED_PATH,
    layer_widths: list[int] | None = None,
    activations: list[str] | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    use_best_found: bool = False,
    nonmetals_only: bool = False,
    log_transform: bool = False,
    optimiser_name: str = "adam",
    seeds: tuple[int, ...] = (42, 43, 44),
    negative_gap_penalty_weight: float = 0.05,
    val_checkpoint: bool = False,
    val_fraction: float = 0.1,
    train_subset: str | None = None,
) -> pd.DataFrame:
    """Run the full OOD benchmark suite, cached under cache/benchmark_results/.

    seeds retrains each fold once per seed, separating split-driven variance from run-to-run
    noise. val_checkpoint gives every fold the best-val-epoch rule the other two models already
    had, and is fingerprinted so both arms coexist.

    train_subset swaps each fold's training side for the "dedup" or "sizematched" arm from
    model.train_subsets, leaving test sides untouched, and is fingerprinted too.

    use_best_found was deprecated on 2026-07-17 and raises, so pass the architecture.
    """
    if layer_widths is None:
        layer_widths = DEFAULT_LAYER_WIDTHS
    if activations is None:
        activations = DEFAULT_ACTIVATIONS
    layer_widths, activations, lr, optimiser_name = resolve_architecture(
        layer_widths, activations, lr, use_best_found, optimiser_name
    )
    settings = {
        "layer_widths": layer_widths, "activations": activations, "epochs": epochs, "lr": lr,
        "batch_size": batch_size,
        "nonmetals_only": nonmetals_only, "log_transform": log_transform,
        "optimiser_name": optimiser_name,
        "seeds": list(seeds),
        "negative_gap_penalty_weight": negative_gap_penalty_weight,
    }
    # Recorded only when on, so pre-2026-08-14 fixed-epoch entries keep their hash.
    if val_checkpoint:
        settings["val_checkpoint"] = True
        settings["val_fraction"] = val_fraction
    # The same only-when-set rule, so the full-data arm keeps its hash.
    if train_subset:
        settings["train_subset"] = train_subset
    # This module is in extra_files, so a split-strategy change invalidates cached results.
    cache_path = get_cache_path(
        "benchmark_results", [data_path], settings,
        extra_files=(_NETWORK_PY, _BENCHMARK_PY, _TRAINING_PY, _FOLDS_PY), ext="csv"
    )
    if os.path.exists(cache_path):
        print(f"Cache valid — loading benchmark results from {cache_path}")
        return pd.read_csv(cache_path)

    if nonmetals_only:
        # The fold artifact indexes the unfiltered file's rows, so dropping metals renumbers them.
        # A filtered variant would need its own artifact.
        raise NotImplementedError(
            "run_benchmark(nonmetals_only=True) is not supported since the OOD folds moved "
            "to the shared artifact (2026-07-27): fold row indices refer to the unfiltered "
            "dataset. No master-plan step uses this option."
        )

    # Through prepare_features, as an inline build_nn_features would key on the wrong file.
    X_df, y_s, formulas_s, _ = prepare_features(data_path)
    y_np = np.asarray(y_s, dtype="float32")
    formulas = np.array(formulas_s.tolist())
    X_raw = X_df.values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  |  {len(y_np)} materials  |  {X_raw.shape[1]} raw features")

    all_rows: list[dict] = []
    pred_frames: list = []     # per-(fold, seed) predictions, written beside the results CSV
    history_frames: list = []  # per-epoch train curves, same

    # Shared across every split below rather than repeated per setting.
    _common = dict(
        log_transform=log_transform,
        optimiser_name=optimiser_name, layer_widths=layer_widths, activations=activations,
        seeds=seeds,
        negative_gap_penalty_weight=negative_gap_penalty_weight,
        val_checkpoint=val_checkpoint,
        val_fraction=val_fraction,
        pred_sink=pred_frames,
        history_sink=history_frames,
        model_dir=cache_path.replace(".csv", "_models"),
    )

    # Folds come from the shared artifact, so identical splits are checkable by checksum.
    # Imported inside the function, as a module-level import would be circular.
    from model.folds import OOD_SUITE_SPLITS, build_ood_folds, load_ood_folds
    from model.train_subsets import apply_subset, resolve_train_subset
    folds_dir = build_ood_folds(data_path)
    entries = load_ood_folds(folds_dir, splits=OOD_SUITE_SPLITS, formulas=formulas)
    # Training side only, every test index stays the artifact's.
    subsets = resolve_train_subset(train_subset, data_path, formulas)
    for i, split_name in enumerate(OOD_SUITE_SPLITS, start=1):
        split_folds = [(apply_subset(subsets, s, lab, tr), te, lab)
                       for s, lab, tr, te in entries if s == split_name]
        if not split_folds:
            print(f"\n=== {i}/{len(OOD_SUITE_SPLITS)}  {split_name} — no folds in artifact, skipping ===")
            continue
        print(f"\n=== {i}/{len(OOD_SUITE_SPLITS)}  {split_name} ({len(split_folds)} folds) ===")
        all_rows += _run_split(split_name, split_folds,
                               X_raw, y_np, device, epochs=epochs, lr=lr, batch_size=batch_size,
                               labeled_folds=True, **_common)

    results_df = pd.DataFrame(all_rows)

    # mean+std across every (fold, seed) row per split, plus a bootstrap CI on MAE and R².
    # No multiple-comparisons correction, so these are a starting point for cross-split claims.
    _metric_cols = ["mae", "rmse", "r2", "mrae", "mae_nonmetals", "rmse_nonmetals", "r2_nonmetals"]
    summary_rows = []
    for split_name, group in results_df.groupby("split"):
        row: dict = {"split": split_name, "n_rows": len(group)}
        for col in _metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std()
        row["mae_ci_lo"], row["mae_ci_hi"] = _bootstrap_ci(group["mae"].to_numpy())
        row["r2_ci_lo"], row["r2_ci_hi"] = _bootstrap_ci(group["r2"].to_numpy())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).set_index("split")

    print("\n\n=== Benchmark Summary (mean±std across folds/seeds; 95% bootstrap CI on MAE/R²) ===")
    for split_name, row in summary.iterrows():
        print(
            f"  {split_name:<16} "
            f"MAE={row['mae_mean']:.4f}±{row['mae_std']:.4f} "
            f"[{row['mae_ci_lo']:.4f}, {row['mae_ci_hi']:.4f}]   "
            f"R²={row['r2_mean']:.4f}±{row['r2_std']:.4f} "
            f"[{row['r2_ci_lo']:.4f}, {row['r2_ci_hi']:.4f}]   "
            f"nonmetals-only MAE={row['mae_nonmetals_mean']:.4f}±{row['mae_nonmetals_std']:.4f}"
        )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    results_df.to_csv(cache_path, index=False)
    write_meta(cache_path, [data_path], settings, extra_files=(_NETWORK_PY, _BENCHMARK_PY, _TRAINING_PY))
    print(f"\nFull results cached to {cache_path}")

    # Written out, as the bootstrap CIs are not bit-reproducible from the rows alone.
    summary.to_csv(cache_path.replace(".csv", "_summary.csv"))

    # Formula included, so the predictions file stands alone without a join by row_idx.
    if pred_frames:
        preds_df = pd.concat(pred_frames, ignore_index=True)
        preds_df.insert(4, "formula", formulas[preds_df["row_idx"].to_numpy()])
        preds_path = cache_path.replace(".csv", "_preds.csv")
        preds_df.to_csv(preds_path, index=False)
        print(f"Per-material predictions cached to {preds_path} ({len(preds_df)} rows)")

    if history_frames:
        hist_path = cache_path.replace(".csv", "_history.csv")
        pd.concat(history_frames, ignore_index=True).to_csv(hist_path, index=False)
        print(f"Per-epoch training curves cached to {hist_path}")

    n_diverged = int(results_df["diverged"].sum()) if "diverged" in results_df else 0
    if n_diverged:
        print(f"\nWARNING: {n_diverged}/{len(results_df)} (fold, seed) runs flagged as diverged "
              f"(R2 < 0 or |pred| > {_DIVERGENCE_PRED_BOUND_EV} eV) -- see the 'diverged' column.")

    return results_df


def _fetch_symmetry_data(material_ids: list[str], cache_path: str) -> pd.DataFrame:
    """Fetch crystal_system and spacegroup_symbol from the MP API and cache to CSV."""
    return _fetch_mp_symmetry(material_ids, cache_path, include_spacegroup=True, label="symmetry data")


_symmetry_data_cache: dict[tuple, pd.DataFrame] = {}


def _load_symmetry_data(
    material_ids: list[str],
    data_path: str,
) -> pd.DataFrame | None:
    """Symmetry data for material_ids, preferring data_path, then RAW_DATA_PATH, then a
    symmetry cache validated by ID overlap. The MP API tops up an existing cache alone, and
    with no cache this prints a preprocessing instruction and returns None."""
    cache_path = get_cache_path("symmetry_data", [data_path], {}, ext="csv")
    key = (tuple(material_ids), cache_path)
    if key in _symmetry_data_cache:
        return _symmetry_data_cache[key]

    _SYM_COLS = {"crystal_system", "spacegroup_symbol"}

    def _from_csv(path: str) -> pd.DataFrame | None:
        if not os.path.exists(path):
            return None
        if not _SYM_COLS.issubset(pd.read_csv(path, nrows=0).columns):
            return None
        sub = pd.read_csv(
            path,
            index_col="material_id",
            usecols=["material_id", "crystal_system", "spacegroup_symbol"],
        ).reindex(material_ids)
        return sub if sub["crystal_system"].notna().any() else None

    result = _from_csv(data_path)
    if result is None:
        result = _from_csv(RAW_DATA_PATH)

    if result is None and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col="material_id")
        overlap = sum(mid in cached.index for mid in material_ids)
        if overlap >= len(material_ids) * 0.1:
            missing = [mid for mid in material_ids if mid not in cached.index]
            if not missing:
                result = cached.reindex(material_ids)
            else:
                print(f"Symmetry cache missing {len(missing)} entries — fetching from MP API...")
                try:
                    new_df = _fetch_symmetry_data(missing, cache_path)
                    combined = pd.concat([new_df, cached])
                    combined = combined.loc[~combined.index.duplicated(keep="first")]
                    combined.to_csv(cache_path)
                    result = combined.reindex(material_ids)
                except Exception as exc:
                    print(f"Warning: symmetry fetch failed ({exc}). Using cached data only.")
                    result = cached.reindex(material_ids)
        else:
            print(f"Warning: symmetry cache ID mismatch ({overlap}/{len(material_ids)} IDs found).")

    if result is None:
        print("Symmetry columns missing — run: python -m data.preprocessing")

    if result is not None:
        _symmetry_data_cache[key] = result
    return result

def evaluate_by_grouping(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    label: str = "Group",
    min_samples: int = 5,
) -> pd.DataFrame:
    """MAE/RMSE/R² per unique value in groups, dropping those below min_samples, plus
    nonmetals-only variants, else groups with different metal fractions differ in MAE and R²
    for that reason alone."""
    rows = []
    for grp in sorted(g for g in set(groups) if pd.notna(g)):
        mask = groups == grp
        if mask.sum() < min_samples:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        row = {
            "class":  grp,
            "n_test": int(mask.sum()),
            "mae":    float(np.abs(yt - yp).mean()),
            "rmse":   float(np.sqrt(((yt - yp) ** 2).mean())),
            "r2":     float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan"),
        }
        nonmetal_mask = yt != 0
        if nonmetal_mask.sum() >= 2:
            yt_nm, yp_nm = yt[nonmetal_mask], yp[nonmetal_mask]
            row["mae_nonmetals"]  = float(np.abs(yt_nm - yp_nm).mean())
            row["rmse_nonmetals"] = float(np.sqrt(((yt_nm - yp_nm) ** 2).mean()))
            row["r2_nonmetals"]   = float(r2_score(yt_nm, yp_nm))
        else:
            row["mae_nonmetals"] = row["rmse_nonmetals"] = row["r2_nonmetals"] = float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"\n--- Test Metrics by {label} ---")
    if not df.empty:
        print(df.to_string(index=False, float_format="{:.4f}".format))
    return df


def evaluate_by_crystal_system(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    material_ids: list[str],
    data_path: str,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Test metrics grouped by crystal system, an empty frame where no symmetry data is available."""
    sym_df = _load_symmetry_data(material_ids, data_path)
    if sym_df is None:
        return pd.DataFrame()
    groups = np.array(sym_df["crystal_system"].tolist(), dtype=object)
    return evaluate_by_grouping(y_true, y_pred, groups, "Crystal System", min_samples)


def evaluate_by_structure_prototype(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    material_ids: list[str],
    data_path: str,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Test metrics grouped by space group symbol, an empty frame where no symmetry data is available."""
    sym_df = _load_symmetry_data(material_ids, data_path)
    if sym_df is None:
        return pd.DataFrame()
    groups = np.array(sym_df["spacegroup_symbol"].tolist(), dtype=object)
    return evaluate_by_grouping(y_true, y_pred, groups, "Space Group", min_samples)


def run_mdhit_comparison(
    raw_path: str = RAW_DATA_PATH,
    filtered_path: str = _DEFAULT_FILTERED_PATH,
    layer_widths: list[int] | None = None,
    activations: list[str] | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    use_best_found: bool = False,
    nonmetals_only: bool = False,
    log_transform: bool = False,
    optimiser_name: str = "adam",
    negative_gap_penalty_weight: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train FlexNet on raw against MD-HIT-filtered data, one row per arm, with no delta
    computed here as the caller compares the two rows.

    use_best_found was deprecated on 2026-07-17 and now raises, so pass the architecture.
    nonmetals_only, log_transform and negative_gap_penalty_weight are described in
    model.training.train, and the penalty is in this comparison's cache key so both arms
    verifiably trained under it.

    random_state applies to both arms, so each CSV is one seed-paired pair of runs, and it
    enters the cache key only where it differs from 42.
    """
    if layer_widths is None:
        layer_widths = DEFAULT_LAYER_WIDTHS
    if activations is None:
        activations = DEFAULT_ACTIVATIONS
    layer_widths, activations, lr, optimiser_name = resolve_architecture(
        layer_widths, activations, lr, use_best_found, optimiser_name
    )
    settings = {
        "layer_widths": layer_widths, "activations": activations, "epochs": epochs, "lr": lr,
        "batch_size": batch_size,
        "nonmetals_only": nonmetals_only, "log_transform": log_transform,
        "optimiser_name": optimiser_name,
        "negative_gap_penalty_weight": negative_gap_penalty_weight,
    }
    # A literal 42, not config.TRAINING_SEEDS[0], as pre-param caches must keep their legacy key.
    # Fold random_state in unconditionally once the legacy §3 artifacts are retired.
    if random_state != 42:
        settings["random_state"] = random_state
    cache_path = get_cache_path(
        "mdhit_comparison", [raw_path, filtered_path], settings,
        extra_files=(_NETWORK_PY, _TRAINING_PY), ext="csv",
    )
    if os.path.exists(cache_path):
        print(f"Cache valid — loading MD-HIT comparison from {cache_path}")
        return pd.read_csv(cache_path)

    rows = []
    for label, path in [("raw", raw_path), ("md_hit_filtered", filtered_path)]:
        print(f"\n{'=' * 60}")
        print(f"MD-HIT comparison: training on {label} data ({path})")
        print(f"{'=' * 60}")
        _, history = train_model(
            data_path=path,
            layer_widths=layer_widths, activations=activations,
            epochs=epochs, lr=lr, batch_size=batch_size,
            nonmetals_only=nonmetals_only,
            log_transform=log_transform,
            optimiser_name=optimiser_name,
            negative_gap_penalty_weight=negative_gap_penalty_weight,
            random_state=random_state,
        )
        rows.append({
            "dataset":  label,
            "random_state": random_state,
            "n_train":  history["n_train"],
            "n_test":   len(history["test_true"]),
            "mae":      history["test_mae"],
            "rmse":     history["test_rmse"],
            "r2":       history["test_r2"],
            "mape":     history.get("test_mape", float("nan")),
        })

    df = pd.DataFrame(rows)
    print("\n\n=== MD-HIT Comparison Summary ===")
    print(df.to_string(index=False, float_format="{:.4f}".format))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    write_meta(cache_path, [raw_path, filtered_path], settings, extra_files=(_NETWORK_PY, _TRAINING_PY))
    print(f"Results cached to {cache_path}")
    return df


def _to_composition(x: Composition | Structure | str) -> Composition:
    """Coerce a matbench task input to a Composition, where str(Structure) is a full lattice
    dump rather than a formula so Structures must go via .composition."""
    if isinstance(x, Composition):
        return x
    if isinstance(x, Structure):
        return x.composition
    return Composition(str(x))


def run_matbench_mp_gap(
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
) -> pd.DataFrame | None:
    """Evaluate FlexNet on the matbench_mp_gap 5-fold benchmark, cached under
    cache/matbench_mp_gap/.

    Dead since 2026-07-16, as matbench is uninstallable here, so a fresh run prints an install
    hint and returns None and model.matbench_folds.run_matbench_from_folds() should be used
    instead. A cache hit still returns a previous run's results.

    log_transform fits log1p(Eg) internally whilst predictions stay in eV.
    """
    if layer_widths is None:
        layer_widths = DEFAULT_LAYER_WIDTHS
    if activations is None:
        activations = DEFAULT_ACTIVATIONS
    layer_widths, activations, lr, optimiser_name = resolve_architecture(
        layer_widths, activations, lr, use_best_found, optimiser_name
    )
    settings = {
        "layer_widths": layer_widths, "activations": activations, "epochs": epochs, "lr": lr,
        "batch_size": batch_size,
        "log_transform": log_transform, "optimiser_name": optimiser_name,
        "random_state": random_state,
        "negative_gap_penalty_weight": negative_gap_penalty_weight,
    }
    cache_path = get_cache_path("matbench_mp_gap", [], settings, extra_files=(_NETWORK_PY,), ext="csv")
    if os.path.exists(cache_path):
        print(f"Cache valid — loading matbench results from {cache_path}")
        return pd.read_csv(cache_path)

    try:
        from matbench.bench import MatbenchBenchmark  # type: ignore[import-untyped]
    except ImportError:
        print("matbench not installed. Run: pip install matbench")
        return None

    mb = MatbenchBenchmark(autoload=False, subset=["matbench_mp_gap"])
    task = mb.matbench_mp_gap
    task.load()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nMatbench mp_gap benchmark  |  device={device}")
    rows: list[dict] = []

    matbench_fold_dir = os.path.join(CACHE_DIR, "matbench_folds")
    os.makedirs(matbench_fold_dir, exist_ok=True)
    for fold_idx in task.folds:
        print(f"\n=== Fold {fold_idx} ===")
        train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
        test_inputs, test_outputs   = task.get_test_data(fold_idx, include_target=True)

        train_comps = [_to_composition(c) for c in train_inputs]
        test_comps  = [_to_composition(c) for c in test_inputs]
        y_train_np  = np.asarray(train_outputs, dtype="float32")
        y_test_np   = np.asarray(test_outputs,  dtype="float32")

        # Save fold CSVs as a stable file for build_nn_features() to hash against
        fold_train_csv = os.path.join(matbench_fold_dir, f"matbench_fold{fold_idx}_train.csv")
        fold_test_csv  = os.path.join(matbench_fold_dir, f"matbench_fold{fold_idx}_test.csv")
        pd.DataFrame({"formula_pretty": [str(c) for c in train_comps], "band_gap": y_train_np}).to_csv(fold_train_csv, index=False)
        pd.DataFrame({"formula_pretty": [str(c) for c in test_comps],  "band_gap": y_test_np }).to_csv(fold_test_csv,  index=False)

        print(f"  Computing features for {len(train_comps)} train + {len(test_comps)} test compositions...")
        X_train_df = build_nn_features(train_comps, fold_train_csv)
        X_test_df  = build_nn_features(test_comps,  fold_test_csv)

        # Concatenate and preprocess jointly so scaler/imputer fit only on train
        X_combined = np.vstack([X_train_df.values, X_test_df.values])
        y_combined = np.concatenate([y_train_np, y_test_np])
        n_tr = len(X_train_df)
        train_idx = np.arange(n_tr)
        test_idx  = np.arange(n_tr, n_tr + len(X_test_df))
        X_tr, X_te = _preprocess_fold(X_combined, y_combined, train_idx, test_idx)

        print(f"  Training: {len(y_train_np)} samples  |  {X_tr.shape[1]} features")
        # Three-tuple unpack, fixed after the extras were added (this path is unused).
        metrics, fold_preds, _ = _train_fold(
            X_tr, y_train_np, X_te, y_test_np, device,
            layer_widths=layer_widths, activations=activations,
            epochs=epochs, lr=lr, batch_size=batch_size,
            log_transform=log_transform,
            optimiser_name=optimiser_name,
            random_state=random_state,
            negative_gap_penalty_weight=negative_gap_penalty_weight,
        )
        print(
            f"  Fold {fold_idx}: MAE={metrics['mae']:.4f} eV  MRAE={metrics['mrae']:.4f}  "
            f"R²={metrics['r2']:.4f}  (nonmetals-only MAE={metrics['mae_nonmetals']:.4f})"
        )
        task.record(fold_idx, pd.Series(fold_preds, index=test_inputs.index))
        rows.append({"fold": fold_idx, "n_train": len(y_train_np), "n_test": len(y_test_np), **metrics})

    results_df = pd.DataFrame(rows)
    summary = results_df[["mae", "rmse", "r2", "mrae", "mae_nonmetals", "rmse_nonmetals", "r2_nonmetals"]].mean()
    print(f"\n=== Matbench mp_gap — Mean across folds ===")
    print(f"  MAE  = {summary['mae']:.4f} eV   (nonmetals-only: {summary['mae_nonmetals']:.4f} eV)")
    print(f"  RMSE = {summary['rmse']:.4f} eV  (nonmetals-only: {summary['rmse_nonmetals']:.4f} eV)")
    print(f"  R²   = {summary['r2']:.4f}       (nonmetals-only: {summary['r2_nonmetals']:.4f})")
    print(f"  MRAE = {summary['mrae']:.4f}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    results_df.to_csv(cache_path, index=False)
    write_meta(cache_path, [], settings, extra_files=(_NETWORK_PY,))
    print(f"\nFull fold results cached to {cache_path}")

    try:
        print("\nMatbench scores:")
        print(mb.get_info())
    except Exception:
        pass

    return results_df



def _unpack_predictions_csv(cache_path: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    df = pd.read_csv(cache_path)
    return df["y_true"].to_numpy(), df["y_pred"].to_numpy(), df["material_id"].tolist(), df["formula"].tolist()


def _load_or_train(
    data_path: str,
    settings: dict,
    *,
    replot: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """(y_true, y_pred, material_ids, formulas), from cache or by training."""
    cache_path = get_cache_path("phase1_predictions", [data_path], settings, extra_files=(_NETWORK_PY,), ext="csv")
    if replot:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"--replot requested but no predictions cache found at {cache_path}. "
                "Run without --replot first."
            )
        print(f"--replot: loading predictions from {cache_path}")
        return _unpack_predictions_csv(cache_path)

    if os.path.exists(cache_path):
        print(f"Predictions cache valid — loading from {cache_path}")
        return _unpack_predictions_csv(cache_path)

    print("No cached predictions — training model...")
    _, history = train_model(data_path=data_path, **{k: settings[k] for k in settings})
    pred_df = pd.DataFrame({
        "material_id": history["test_material_ids"],
        "formula":     history["test_formulas"],
        "y_true":      history["test_true"].tolist(),
        "y_pred":      history["test_pred"].tolist(),
    })
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pred_df.to_csv(cache_path, index=False)
    write_meta(cache_path, [data_path], settings, extra_files=(_NETWORK_PY,))
    print(f"Predictions cached to {cache_path}")
    return (
        history["test_true"],
        history["test_pred"],
        history["test_material_ids"],
        history["test_formulas"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OOD benchmark + Phase 1 evaluation for FlexNet")
    parser.add_argument("--data-file",       default=_DEFAULT_FILTERED_PATH)
    parser.add_argument("--raw-data-file",   default=RAW_DATA_PATH)
    parser.add_argument("--layer-widths",    type=int, nargs="+", default=DEFAULT_LAYER_WIDTHS,
                        help="One width per hidden layer")
    parser.add_argument("--activations",     nargs="+", choices=list(FLEX_ACTIVATIONS), default=DEFAULT_ACTIVATIONS,
                        help="One activation per hidden layer, same length as --layer-widths")
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch-size",      type=int,   default=256)
    parser.add_argument("--phase1-only",     action="store_true",
                        help="Run only Phase 1 evaluation (skip OOD benchmark suite)")
    parser.add_argument("--ood-only",        action="store_true",
                        help="Run only the OOD benchmark suite (skip Phase 1 evaluation, "
                             "MD-HIT comparison, and Matbench mp_gap)")
    parser.add_argument("--skip-matbench",   action="store_true")
    parser.add_argument("--skip-mdhit",      action="store_true")
    parser.add_argument("--optimiser",       choices=list(OPTIMISER_CHOICES), default="adam",
                        help="Used only for --ood-only/run_benchmark (Phase 1/MD-HIT/Matbench "
                             "steps above always use \"adam\")")
    parser.add_argument("--replot",          action="store_true",
                        help="Phase 1 only: load saved predictions and regenerate plots "
                             "instead of training. The MD-HIT, Matbench and OOD steps "
                             "still train as normal.")
    parser.add_argument("--nonmetals-only",  action="store_true",
                        help="Drop Eg=0 metals before training/evaluation across all steps below "
                             "(BGML-style nonmetal-only regression)")
    parser.add_argument("--log-transform",   action="store_true",
                        help="Fit against log1p(Eg) instead of raw eV across all steps below "
                             "(reported metrics stay in eV)")
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 43, 44],
                        help="Seed(s) for weight init + DataLoader shuffle (see _train_fold). "
                             "run_benchmark() retrains every fold once per seed given (see "
                             "_run_split) -- pass a single value to train once per fold. "
                             "run_matbench_mp_gap() uses only the first.")
    parser.add_argument("--show",            action="store_true", dest="show_figures")
    args = parser.parse_args()

    if args.phase1_only and args.ood_only:
        parser.error("--phase1-only and --ood-only are mutually exclusive")

    if len(args.activations) != len(args.layer_widths):
        parser.error(
            f"--activations must be given with one entry per --layer-widths entry "
            f"({len(args.layer_widths)}), got {args.activations!r}"
        )

    kwargs = dict(
        layer_widths=args.layer_widths,
        activations=args.activations,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        nonmetals_only=args.nonmetals_only,
        log_transform=args.log_transform,
    )

    if not args.ood_only:
        print("\n" + "=" * 60)
        print("Phase 1: Standard model evaluation (70/15/15 split)")
        print("=" * 60)
        y_true, y_pred, material_ids, _ = _load_or_train(
            args.data_file, kwargs, replot=args.replot
        )

        from visualisation import (
            plot_error_distribution,
            plot_metrics_by_class,
            plot_parity_by_gap_magnitude,
        )
        plot_error_distribution(y_true, y_pred)
        plot_parity_by_gap_magnitude(y_true, y_pred)

        cs_df = evaluate_by_crystal_system(y_true, y_pred, material_ids, data_path=args.data_file)
        if not cs_df.empty:
            plot_metrics_by_class(cs_df, title="Test Metrics by Crystal System", filename="metrics_by_crystal_system.svg")

        proto_df = evaluate_by_structure_prototype(y_true, y_pred, material_ids, data_path=args.data_file)
        if not proto_df.empty:
            plot_metrics_by_class(proto_df, title="Test Metrics by Structure Prototype", filename="metrics_by_structure_prototype.svg")

        if not args.skip_mdhit:
            mdhit_df = run_mdhit_comparison(
                raw_path=args.raw_data_file,
                filtered_path=args.data_file,
                **kwargs,
            )
            from visualisation import plot_mdhit_comparison
            plot_mdhit_comparison(mdhit_df)

        # nonmetals_only is not applied, as matbench_mp_gap has fixed predetermined folds.
        if not args.skip_matbench:
            run_matbench_mp_gap(
                random_state=args.seeds[0],
                **{k: v for k, v in kwargs.items() if k != "nonmetals_only"},
            )

    if not args.phase1_only:
        run_benchmark(
            data_path=args.data_file, optimiser_name=args.optimiser,
            seeds=tuple(args.seeds),
            **kwargs,
        )

    if args.show_figures:
        import matplotlib.pyplot as plt
        plt.show()
