"""Exports the exact train/val/test rows a model.training.train() call would use, as three
(formula, target, material_id) CSVs under cache/exported_splits/.

External baselines have to train on the same compositions in the same partitions as FlexNet,
else an accuracy difference confounds architecture with the draw of the split. sklearn's
train_test_split shuffles on row count and random_state alone, so no feature matrix is needed
here.

Run via `python -m model.export_splits` or an "export_splits" plan step.
"""
import os

import numpy as np
import pandas as pd

from analysis import clean_data, filter_nonmetals, load_data
from cache import CACHE_DIR, data_stem, fingerprint_valid, write_meta
from data.preprocessing import default_filtered_data_path

from config import PINNED_DATASET as _DEFAULT_FILTERED_PATH
from model.training import _train_val_test_split

_EXPORT_SPLITS_PY = os.path.abspath(__file__)


def export_splits(
    data_path: str = _DEFAULT_FILTERED_PATH,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    nonmetals_only: bool = False,
) -> dict:
    """Write train/val/test CSVs reproducing train()'s exact split for these arguments.

    Returns the three paths plus n_train/n_val/n_test. An existing export with a matching
    fingerprint is returned as-is.
    """
    settings = {
        "val_size": val_size, "test_size": test_size, "random_state": random_state,
        "nonmetals_only": nonmetals_only,
    }
    subdir = os.path.join(
        CACHE_DIR, "exported_splits",
        f"{data_stem(data_path)}_seed{random_state}" + ("_nonmetals" if nonmetals_only else ""),
    )
    paths = {part: os.path.join(subdir, f"{part}.csv") for part in ("train", "val", "test")}

    extra = (_EXPORT_SPLITS_PY,)
    if all(fingerprint_valid(p, [data_path], settings, extra_files=extra) for p in paths.values()):
        print(f"Cache valid — exported splits already at {subdir}")
        counts = {f"n_{part}": len(pd.read_csv(p)) for part, p in paths.items()}
        return {**paths, **counts}

    df = load_data(data_path)
    valid = clean_data(df)
    if nonmetals_only:
        valid = filter_nonmetals(valid)

    y_np = np.asarray(valid["band_gap"].reset_index(drop=True), dtype="float32")
    formulas = np.array(valid["formula_pretty"].reset_index(drop=True).tolist())
    material_ids = np.array(valid.index.to_series().reset_index(drop=True).tolist())

    # The same helper train() uses, so identical (n, sizes, seed) give identical partitions.
    # X never influences sklearn's shuffle, so a placeholder column is enough.
    placeholder = np.zeros((len(y_np), 1), dtype="float32")
    _, _, _, y_train, y_val, y_test, idx_train, idx_test = _train_val_test_split(
        placeholder, y_np, val_size, test_size, random_state
    )
    idx_all = np.arange(len(y_np))
    idx_val = np.setdiff1d(idx_all, np.concatenate([idx_train, idx_test]))

    os.makedirs(subdir, exist_ok=True)
    for part, idx in (("train", idx_train), ("val", idx_val), ("test", idx_test)):
        out = pd.DataFrame({
            "formula": formulas[idx],
            "target": y_np[idx],
            "material_id": material_ids[idx],
        })
        out.to_csv(paths[part], index=False)
        write_meta(paths[part], [data_path], settings, extra_files=extra)
        print(f"  {part}: {len(out)} rows -> {paths[part]}")

    return {**paths, "n_train": len(idx_train), "n_val": len(idx_val), "n_test": len(idx_test)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=_DEFAULT_FILTERED_PATH)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42, dest="random_state")
    parser.add_argument("--nonmetals-only", action="store_true")
    args = parser.parse_args()

    result = export_splits(
        data_path=args.data_file, val_size=args.val_size, test_size=args.test_size,
        random_state=args.random_state, nonmetals_only=args.nonmetals_only,
    )
    print(result)
