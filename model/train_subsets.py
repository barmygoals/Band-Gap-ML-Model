"""Training-side subsets of the shared OOD folds, MD-HIT de-duplication and its
size-matched random control, computed once and read by all three model runners.

The training side is de-duplicated and the evaluation side never is, so each fold keeps its
test indices byte-identical to the published runs (DECISIONS.md 2026-08-14). The size-matched
control exists so better generalisation cannot be confused with the effect of less data.

Separate from model/folds.py, whose content keys every OOD result.

Run out of chain, then md5-gate it in phase 0:
    python -m model.train_subsets [--threshold 0.5] [--np 10]
"""
import argparse
import json
import os
import zlib

import numpy as np
import pandas as pd

from cache import get_cache_path, write_meta
from config import PINNED_DATASET
from model.folds import SPLIT_ORDER, build_ood_folds, load_ood_folds
from model.training import prepare_features

# _elmd_filter rather than preprocess(), as it runs on a fold's rows rather than a whole CSV.
from data.preprocessing import _elmd_filter

_TRAIN_SUBSETS_PY = os.path.abspath(__file__)

ARMS = ("dedup", "sizematched")
DEFAULT_THRESHOLD = 0.5
DEFAULT_SIMILARITY = "mod_petti"  # mod_petti and fast are the only numba-free metrics (see patch_elmd.py)


def _rng_for(seed: int, split: str, fold: str) -> np.random.Generator:
    """Per-fold generator seeded from (seed, split, fold), by crc32 rather than hash() as
    Python's hash is salted per process, and per fold so adding or reordering splits cannot
    change the draw for folds that already exist."""
    return np.random.default_rng([seed, zlib.crc32(f"{split}/{fold}".encode())])


def build_train_subsets(
    data_path: str = PINNED_DATASET,
    threshold: float = DEFAULT_THRESHOLD,
    similarity: str = DEFAULT_SIMILARITY,
    seed: int = 42,
    n_processes: int = 10,
    threshold_ev: float = 4.0,
    resume: bool = True,
) -> str:
    """Build the per-fold training subsets into a cached directory and return its path, where
    the MD-HIT run per fold depends on that fold's training formulas alone and resume=True
    restarts a wall-killed job at the split it died on."""
    folds_dir = build_ood_folds(data_path, threshold_ev=threshold_ev, seed=seed)
    manifest_in = os.path.join(folds_dir, "manifest.json")

    settings = {
        "threshold": threshold, "similarity": similarity, "seed": seed,
        "folds_dir": os.path.basename(folds_dir),
    }
    # Keyed on the fold manifest, so a fold rebuild invalidates these subsets.
    stem = get_cache_path("ood_train_subsets", [manifest_in], settings,
                          extra_files=(_TRAIN_SUBSETS_PY,), ext="csv")
    out_dir = os.path.splitext(stem)[0]
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path):
        print(f"Cache valid — loading OOD training subsets from {out_dir}")
        return out_dir

    X_df, y_s, formulas_s, _ = prepare_features(data_path)
    formulas = np.array(formulas_s.tolist())
    entries = load_ood_folds(folds_dir, formulas=formulas)
    os.makedirs(out_dir, exist_ok=True)

    by_split: dict[str, list] = {}
    for split, fold, tr_idx, _te in entries:
        by_split.setdefault(split, []).append((fold, tr_idx))

    summary: dict[str, list] = {}
    for split in SPLIT_ORDER:
        folds = by_split.get(split, [])
        split_csv = os.path.join(out_dir, f"{split}.csv")
        if resume and os.path.exists(split_csv):
            done = pd.read_csv(split_csv)
            print(f"  {split:<16} resumed from cache ({len(done)} rows)")
            summary[split] = _summarise(done, folds)
            continue
        frames = []
        rows_note = []
        for fold, tr_idx in folds:
            sub = pd.DataFrame({"formula": formulas[tr_idx]})
            kept = _elmd_filter(sub, threshold, "formula", similarity, n_processes)
            # Row-level from a formula-level decision, as polymorphs share a formula.
            mask = np.array([f in kept for f in formulas[tr_idx]])
            dedup_idx = np.asarray(tr_idx)[mask]
            # Size-matched at the same count, so the arms differ in which rows rather than how many.
            rng = _rng_for(seed, split, fold)
            sm_idx = rng.choice(np.asarray(tr_idx), size=len(dedup_idx), replace=False)
            sm_idx.sort()
            for arm, idx in (("dedup", dedup_idx), ("sizematched", sm_idx)):
                frames.append(pd.DataFrame({
                    "fold": fold, "arm": arm, "row_idx": idx, "formula": formulas[idx],
                }))
            rows_note.append((fold, len(tr_idx), len(dedup_idx)))
            print(f"  {split}/{fold}: train {len(tr_idx)} -> dedup {len(dedup_idx)} "
                  f"({len(dedup_idx) / max(len(tr_idx), 1):.1%}) + size-matched {len(sm_idx)}")
        df = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=["fold", "arm", "row_idx", "formula"]))
        df.to_csv(split_csv, index=False)
        summary[split] = [{"fold": f, "n_train": n, "n_dedup": d} for f, n, d in rows_note]

    manifest = {
        "data_path": data_path, "folds_dir": folds_dir, "settings": settings,
        "arms": list(ARMS), "splits": summary,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    write_meta(manifest_path, [manifest_in], settings,
               extra_files=(_TRAIN_SUBSETS_PY,), name="ood_train_subsets")
    print(f"\nTraining subsets -> {out_dir}")
    return out_dir


def _summarise(df: pd.DataFrame, folds: list) -> list[dict]:
    """Fold sizes for a resumed split, rebuilt from the CSV so the manifest is the same either way."""
    sizes = {str(f): len(idx) for f, idx in folds}
    out = []
    for fold, grp in df.groupby("fold", sort=False):
        n_dedup = int((grp.arm == "dedup").sum())
        out.append({"fold": str(fold), "n_train": sizes.get(str(fold)), "n_dedup": n_dedup})
    return out


def load_train_subsets(
    subsets_dir: str,
    arm: str,
    formulas: np.ndarray | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    """Read one arm back as {(split, fold): train_row_idx}, where formulas asserts the caller's
    row ordering against the artifact's as in model.folds.load_ood_folds."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    out: dict[tuple[str, str], np.ndarray] = {}
    for split in SPLIT_ORDER:
        path = os.path.join(subsets_dir, f"{split}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if not len(df):
            continue
        df = df[df.arm == arm]
        if formulas is not None and len(df):
            got = formulas[df["row_idx"].to_numpy()]
            if not (got == df["formula"].to_numpy()).all():
                raise RuntimeError(
                    f"ROW ORDER MISMATCH in {path}: the caller's row ordering does not "
                    "match the training-subset artifact's. These are row indices, so "
                    "training with this ordering would fit the wrong materials."
                )
        for fold, grp in df.groupby("fold", sort=False):
            out[(split, str(fold))] = grp["row_idx"].to_numpy()
    return out


def resolve_train_subset(
    arm: str | None,
    data_path: str,
    formulas: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    similarity: str = DEFAULT_SIMILARITY,
    seed: int = 42,
    threshold_ev: float = 4.0,
) -> dict[tuple[str, str], np.ndarray] | None:
    """None for arm=None, the full-data arm, and otherwise that arm's lookup, building the
    artifact where it is not cached. The single entry point the three OOD runners call."""
    if arm is None:
        return None
    subsets_dir = build_train_subsets(
        data_path=data_path, threshold=threshold, similarity=similarity,
        seed=seed, threshold_ev=threshold_ev,
    )
    return load_train_subsets(subsets_dir, arm, formulas=formulas)


def apply_subset(
    lookup: dict[tuple[str, str], np.ndarray] | None,
    split: str,
    fold: str,
    train_idx: np.ndarray,
) -> np.ndarray:
    """train_idx, replaced by this fold's subset when one is in force, where a missing entry
    raises rather than passing through, else a dedup arm would be partly the full-data arm."""
    if lookup is None:
        return train_idx
    key = (split, str(fold))
    if key not in lookup:
        raise KeyError(
            f"no training subset for {split}/{fold} in the artifact. Rebuild it with "
            "`uv run python -m model.train_subsets` -- refusing to fall back to the full "
            "training set, which would silently mix arms."
        )
    return lookup[key]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default=PINNED_DATASET)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--similarity", default=DEFAULT_SIMILARITY)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--np", type=int, default=10, dest="n_processes")
    p.add_argument("--no-resume", action="store_true",
                   help="recompute every split instead of keeping finished ones")
    a = p.parse_args()
    build_train_subsets(a.data_path, threshold=a.threshold, similarity=a.similarity,
                        seed=a.seed, n_processes=a.n_processes, resume=not a.no_resume)
