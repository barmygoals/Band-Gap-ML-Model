"""The definition of every OOD fold this project evaluates, frozen into one artifact.

Fold construction used to live in three modules and had drifted apart, so the three models
were compared on folds that were not provably the same (DECISIONS.md 2026-07-27). Building
them once into an md5-gateable directory makes identical splits checkable by checksum.

Layout, under cache/ood_folds/<hash>/:
    <split>.csv     fold, role, row_idx, formula   (role in {train, test})
    manifest.json   settings, per-split counts, explicit skip records

Split constructors are imported from model.benchmark and model.extrapolation, never
re-implemented here. Seven splits are written, the six of OOD_SUITE_SPLITS together with
extrapolation, which has its own step. feature_ood is excluded by design, with
EXCLUDED_SPLITS carrying the reason into every manifest.
"""
import json
import os
import shutil

import numpy as np
import pandas as pd

from analysis import clean_data, load_data
from cache import get_cache_path, write_meta
from config import PINNED_DATASET
from model.benchmark import (
    _MATFOLD_TMP_DIR,
    _build_matfold,
    _load_crystal_systems,
    chemical_system_splits,
    classify_material,
    crystal_system_splits,
    data_efficiency_splits,
    lomo_splits,
    periodic_group_splits,
    random_stratified_split,
)
from model.extrapolation import gap_extrapolation_folds
from model.training import prepare_features

_FOLDS_PY = os.path.abspath(__file__)
_BENCHMARK_PY = os.path.join(os.path.dirname(_FOLDS_PY), "benchmark.py")
_EXTRAPOLATION_PY = os.path.join(os.path.dirname(_FOLDS_PY), "extrapolation.py")

# Written in the order the results chapter reads them rather than alphabetically.
SPLIT_ORDER = (
    "random", "lomo", "chemical_system",
    "periodic_group", "crystal_system", "data_efficiency", "extrapolation",
)

# The OOD suite proper, which is what run_benchmark evaluates.
# extrapolation shares the artifact so all three models get one definition of its controls.
# It is run by its own step.
OOD_SUITE_SPLITS = tuple(s for s in SPLIT_ORDER if s != "extrapolation")

# Recorded in every manifest, so the exclusion reason is in the artifact and not the run log.
# Not runtime skips, as the code path is never entered.
EXCLUDED_SPLITS = {
    "feature_ood": (
        "Excluded 2026-07-27. Wang et al. define this split by k-means (k=8) over a "
        "CGCNN-derived STRUCTURE embedding. It is the only split in the suite defined by "
        "distance in the model's own feature space rather than by a property of the "
        "materials, so it is representation-relative and does not transfer to a "
        "composition-only pipeline. Concretely, on 118 one-hot element-fraction columns "
        "an element occurring in exactly one compound standardises to z = sqrt(N) "
        "(sqrt(32585) = 180.5, measured 180.51 for Ne), and k-means isolates it as soon "
        "as k is large enough to spare a cluster -- so the farthest cluster is a single "
        "monatomic noble-gas solid. Wang's 0.5-5 eV filter removes all five noble gases "
        "(6.2-17.6 eV), so their pipeline cannot encounter this. Re-tuning k was rejected: "
        "every viable k was reachable only by inspecting our own cluster compositions, so "
        "any selection rule would be reverse-engineered from the outcome, and the "
        "resulting clusters are either 90-97% metallic (k=6/7/12, measuring metal "
        "detection) or wide-gap (k=5/9, overlapping the extrapolation test). The other six "
        "splits are defined on material properties and transfer unchanged. See "
        "docs/BENCHMARK_FIDELITY.md and DECISIONS.md 2026-07-27."
    ),
}


def _normalise(folds: list) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Accept plain (train, test) pairs and labelled (train, test, label) triples alike and emit
    (label, train, test), numbering unlabelled folds by position so fold names stay comparable
    with the published results CSVs."""
    out = []
    for i, fold in enumerate(folds):
        if len(fold) == 3:
            train_idx, test_idx, label = fold
        else:
            train_idx, test_idx = fold
            label = str(i)
        out.append((str(label), np.asarray(train_idx), np.asarray(test_idx)))
    return out


def build_ood_folds(
    data_path: str = PINNED_DATASET,
    threshold_ev: float = 4.0,
    seed: int = 42,
    n_splits: int = 10,
) -> str:
    """Build every evaluated OOD fold into a cached directory and return its path, where seed
    fixes the stratified draws and the extrapolation control membership rather than any
    initialisation, which every model varies over config.TRAINING_SEEDS."""
    settings = {
        "threshold_ev": threshold_ev, "seed": seed, "n_splits": n_splits,
        "excluded_splits": sorted(EXCLUDED_SPLITS),
    }
    extra = (_FOLDS_PY, _BENCHMARK_PY, _EXTRAPOLATION_PY)
    stem = get_cache_path("ood_folds", [data_path], settings, extra_files=extra, ext="csv")
    folds_dir = os.path.splitext(stem)[0]
    manifest_path = os.path.join(folds_dir, "manifest.json")
    if os.path.exists(manifest_path):
        print(f"Cache valid — loading OOD folds from {folds_dir}")
        return folds_dir

    X_df, y_s, formulas_s, _ = prepare_features(data_path)
    # float32 matches run_gap_extrapolation's historical cast, as in gap_extrapolation_folds
    y = np.asarray(y_s, dtype="float32")
    formulas = np.array(formulas_s.tolist())
    indices = np.arange(len(y))
    categories = np.array([classify_material(f) for f in formulas])
    print(f"Building OOD folds on {data_path}: {len(y)} rows")

    built: dict[str, list] = {}
    skips: dict[str, str] = {}

    built["random"] = _normalise(random_stratified_split(indices, categories, random_state=seed))
    built["lomo"] = _normalise(lomo_splits(indices, categories))

    shutil.rmtree(_MATFOLD_TMP_DIR, ignore_errors=True)
    print("Building MatFold grouping (chemsys / periodic-table-group)...")
    mf = _build_matfold(indices, formulas)
    built["chemical_system"] = _normalise(chemical_system_splits(mf, n_splits=n_splits))
    built["periodic_group"] = _normalise(periodic_group_splits(mf, n_splits=n_splits))

    # Real MP symmetry, as MatFold's crystalsys reads the dummy structures (BENCHMARK_FIDELITY.md).
    crystal_systems = _load_crystal_systems(clean_data(load_data(data_path)), data_path)
    if crystal_systems is None:
        built["crystal_system"] = []
        skips["crystal_system"] = "crystal_system metadata unavailable (no column, no cache, MP fetch failed)"
    else:
        built["crystal_system"] = _normalise(crystal_system_splits(indices, crystal_systems))

    built["data_efficiency"] = _normalise(
        data_efficiency_splits(indices, categories, random_state=seed)
    )
    built["extrapolation"] = _normalise(
        [(tr, te, name) for name, tr, te in gap_extrapolation_folds(y, threshold_ev, seed=seed)]
    )

    os.makedirs(folds_dir, exist_ok=True)
    manifest: dict = {"data_path": data_path, "n_rows": int(len(y)),
                      "settings": settings, "splits": {},
                      "excluded_splits": EXCLUDED_SPLITS}
    for split in SPLIT_ORDER:
        folds = built.get(split, [])
        frames = []
        for label, train_idx, test_idx in folds:
            for role, idx in (("train", train_idx), ("test", test_idx)):
                frames.append(pd.DataFrame({
                    "fold": label, "role": role,
                    "row_idx": idx, "formula": formulas[idx],
                }))
        df = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=["fold", "role", "row_idx", "formula"]))
        df.to_csv(os.path.join(folds_dir, f"{split}.csv"), index=False)
        manifest["splits"][split] = {
            "n_folds": len(folds),
            "folds": [{"fold": lab, "n_train": int(len(tr)), "n_test": int(len(te))}
                      for lab, tr, te in folds],
            **({"skipped": skips[split]} if split in skips else {}),
        }
        note = f"  SKIPPED — {skips[split]}" if split in skips else ""
        print(f"  {split:<16} {len(folds):>3} folds{note}")

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    write_meta(manifest_path, [data_path], settings, extra_files=extra, name="ood_folds")

    total = sum(m["n_folds"] for m in manifest["splits"].values())
    print(f"\n{total} folds across {len(SPLIT_ORDER)} splits -> {folds_dir}")
    return folds_dir


def load_ood_folds(
    folds_dir: str,
    splits: tuple[str, ...] | None = None,
    formulas: np.ndarray | None = None,
) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Read an artifact back as (split, fold, train_idx, test_idx) tuples, where splits
    restricts the read and formulas should be passed, as it asserts the caller's row ordering
    against the artifact's."""
    out: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for split in (splits or SPLIT_ORDER):
        path = os.path.join(folds_dir, f"{split}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if not len(df):
            continue
        if formulas is not None:
            got = formulas[df["row_idx"].to_numpy()]
            if not (got == df["formula"].to_numpy()).all():
                raise RuntimeError(
                    f"ROW ORDER MISMATCH in {path}: the caller's row ordering does not "
                    "match the fold artifact's. Folds are row-index based, so evaluating "
                    "with this ordering would silently score the wrong materials."
                )
        for label, grp in df.groupby("fold", sort=False):
            out.append((
                split, str(label),
                grp.loc[grp.role == "train", "row_idx"].to_numpy(),
                grp.loc[grp.role == "test", "row_idx"].to_numpy(),
            ))
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default=PINNED_DATASET)
    p.add_argument("--threshold-ev", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    build_ood_folds(a.data_path, threshold_ev=a.threshold_ev, seed=a.seed)
