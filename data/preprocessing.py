#Composition-based redundancy-reduction methodology adapted from:
#Q.Li, N.Fu, S.Omee, J.Hu, MD-HIT: MACHINE LEARNING FOR MATERIALS PROPERTY PREDICTION WITH DATASET REDUNDANCY CONTROL. Arxiv 2023.6
#github.com/usccolumbia/MD-HIT
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

from cache import get_cache_path, write_meta
from config import ELMD_DISTANCE_THRESHOLD, MAX_ENERGY_ABOVE_HULL, RAW_DATA_PATH, REQUIRED_COLUMNS
from data.dataset_analysis import _reduced_formula

_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external_libraries",
    "MD_hit_formula_parallel.py",
)
# In both cache calls' extra_files, else filter edits leave stale CSVs.
_PREPROCESSING_PY = os.path.abspath(__file__)


def _dedupe_by_composition(df: pd.DataFrame, prefer_min_col: str | None = None) -> pd.DataFrame:
    """Collapse to one row per parsed reduced composition, keeping the lowest prefer_min_col
    value where that column exists and the first-seen row otherwise. NaNs sort last and
    kind="stable" keeps ties deterministic."""
    df = df.assign(_composition=df["formula_pretty"].apply(_reduced_formula))
    df = df.dropna(subset=["_composition"])
    if prefer_min_col is not None and prefer_min_col in df.columns:
        df = df.sort_values(prefer_min_col, na_position="last", kind="stable")
    deduped = df.drop_duplicates(subset="_composition", keep="first")
    return deduped.drop(columns=["_composition"])


def _elmd_filter(
    df: pd.DataFrame,
    threshold: float,
    formula_column: str,
    similarity: str,
    n_processes: int,
) -> set[str]:
    """Deduplicate by ElMD distance through the MD_hit_formula_parallel.py subprocess, returning the kept formula strings."""
    tmpdir = tempfile.mkdtemp()
    try:
        input_csv = os.path.join(tmpdir, "input.csv")
        outfile_prefix = os.path.join(tmpdir, "out")

        df[[formula_column]].drop_duplicates().to_csv(input_csv, index=False)

        subprocess.run(
            [
                sys.executable, _SCRIPT_PATH,
                "--inputfile", input_csv,
                "--threshold", str(threshold),
                "--formula_column", formula_column,
                "--outfile", outfile_prefix,
                "--similarity", similarity,
                "--np", str(n_processes),
            ],
            check=True,
        )

        matches = glob.glob(os.path.join(tmpdir, "out_formulas_nr_*.csv"))
        if not matches:
            raise RuntimeError(f"MD-HIT script ran but produced no output CSV in {tmpdir}")

        kept_df = pd.read_csv(matches[0])
        return set(kept_df["formula"].dropna().tolist())
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def default_filtered_data_path(
    input_path: str = RAW_DATA_PATH,
    threshold: float = ELMD_DISTANCE_THRESHOLD,
    similarity: str = "mod_petti",
) -> str:
    """The cache path preprocess() would use for these settings, without running it. Other
    modules no longer call it for their default data path, as they use config.PINNED_DATASET."""
    settings = {"threshold": threshold, "similarity": similarity}
    return get_cache_path(
        "filtered_data", [input_path], settings, extra_files=(_SCRIPT_PATH, _PREPROCESSING_PY)
    )


def preprocess(
    input_path: str,
    threshold: float = ELMD_DISTANCE_THRESHOLD,
    similarity: str = "mod_petti",  # mod_petti and fast are the only numba-free metrics (Python 3.14+)
    n_processes: int = 10,
) -> str:
    """Remove compositionally similar entries via ElMD distance (MD-HIT), then dedupe to one
    row per composition, returning the filtered CSV's path. The second step is not redundant,
    as without it every polymorph row of a surviving formula is kept and one composition can
    land on both sides of a split."""
    settings = {"threshold": threshold, "similarity": similarity}
    output_path = default_filtered_data_path(input_path, threshold, similarity)
    if os.path.exists(output_path):
        print(f"Cache valid — skipping preprocessing ({output_path})")
        return output_path

    df = pd.read_csv(input_path, index_col="material_id")
    print(f"Loaded {len(df)} materials from {input_path}")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    print(f"{len(df)} materials remaining after dropping nulls")

    kept_formulas = _elmd_filter(df, threshold, "formula_pretty", similarity, n_processes)

    removed = df["formula_pretty"].nunique() - len(kept_formulas)
    print(f"Composition filter: kept {len(kept_formulas)} / {df['formula_pretty'].nunique()} unique formulas, removed {removed} (threshold={threshold})")

    filtered = df[df["formula_pretty"].isin(kept_formulas)]
    print(f"{len(filtered)} materials retained after composition deduplication")

    deduped = _dedupe_by_composition(filtered, prefer_min_col="energy_above_hull")
    print(
        f"{len(deduped)} materials retained after collapsing to one row per composition "
        f"({len(filtered)} before this step, {len(filtered) - len(deduped)} duplicate "
        f"polymorph rows removed)"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    deduped.to_csv(output_path, index=True)
    write_meta(output_path, [input_path], settings, extra_files=(_SCRIPT_PATH, _PREPROCESSING_PY))
    print(f"Filtered dataset saved to {output_path}")
    return output_path


def _stability_cache_inputs(input_path: str, hull_source_path: str) -> list[str]:
    """[input_path], plus hull_source_path where input_path lacks energy_above_hull, from a
    header-only peek so both callers agree on the cache key's shape without loading the CSV."""
    has_hull_col = "energy_above_hull" in pd.read_csv(input_path, nrows=0).columns
    return [input_path] if has_hull_col else [input_path, hull_source_path]


def default_stability_filtered_path(
    input_path: str = RAW_DATA_PATH,
    max_e_above_hull: float = MAX_ENERGY_ABOVE_HULL,
    hull_source_path: str = RAW_DATA_PATH,
) -> str:
    """The cache path filter_by_stability() would use, without running it, in its own namespace since stability and MD-HIT filtering are independent axes."""
    settings = {"max_e_above_hull": max_e_above_hull}
    return get_cache_path(
        "stability_filtered", _stability_cache_inputs(input_path, hull_source_path), settings,
        extra_files=(_PREPROCESSING_PY,),
    )


def filter_rare_elements(
    input_path: str,
    min_element_count: int = 6,
    formula_column: str = "formula_pretty",
) -> str:
    """Drop every composition holding an element in fewer than `min_element_count`
    compositions of `input_path`, and return the filtered CSV's path. An element seen in a
    handful of compositions cannot be learned, so those rows are dropped rather than left
    unanchored. Not iterated to a fixed point, since a moving criterion would depend on
    removal order (supervisor request, DECISIONS.md 2026-08-15)."""
    settings = {"min_element_count": min_element_count, "formula_column": formula_column}
    output_path = get_cache_path(
        "element_filtered", [input_path], settings, extra_files=(_PREPROCESSING_PY,)
    )
    if os.path.exists(output_path):
        print(f"Cache valid — skipping rare-element filtering ({output_path})")
        return output_path

    from collections import Counter
    from pymatgen.core import Composition

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    # Raw exports carry null formulas, which the pinned dataset does not.
    # Dropped rather than raising, as a null formula cannot survive an element filter anyway.
    n_before = len(df)
    df = df[df[formula_column].notna()].reset_index(drop=True)
    if len(df) < n_before:
        print(f"  dropped {n_before - len(df)} row(s) with a null {formula_column}")
    el_sets = [frozenset(Composition(f).get_el_amt_dict().keys()) for f in df[formula_column]]
    counts = Counter(e for s in el_sets for e in s)
    rare = {e for e, c in counts.items() if c < min_element_count}
    keep = [not (s & rare) for s in el_sets]

    kept_elements = sorted({e for s, k in zip(el_sets, keep) if k for e in s})
    print(f"  elements appearing in < {min_element_count} compositions ({len(rare)}): "
          f"{', '.join(sorted(rare, key=lambda e: counts[e]))}")
    print(f"  counts: {[counts[e] for e in sorted(rare, key=lambda e: counts[e])]}")
    print(f"  dropping {len(df) - sum(keep)} composition(s); {sum(keep)} remain")
    print(f"  element vocabulary: {len(counts)} -> {len(kept_elements)}")

    out = df[keep]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    write_meta(output_path, [input_path], settings, extra_files=(_PREPROCESSING_PY,))
    print(f"Element-filtered data saved to {output_path}")
    return output_path


def filter_by_stability(
    input_path: str = RAW_DATA_PATH,
    max_e_above_hull: float = MAX_ENERGY_ABOVE_HULL,
    hull_source_path: str = RAW_DATA_PATH,
) -> str:
    """Keep materials with energy_above_hull <= max_e_above_hull, then dedupe to one row per
    reduced composition keeping the lowest-hull polymorph, returning the filtered CSV's path
    and never overwriting input_path. hull_source_path backfills energy_above_hull where
    input_path predates that column, which is valid as filtering only removes rows."""
    settings = {"max_e_above_hull": max_e_above_hull}
    cache_inputs = _stability_cache_inputs(input_path, hull_source_path)
    output_path = get_cache_path(
        "stability_filtered", cache_inputs, settings, extra_files=(_PREPROCESSING_PY,)
    )
    if os.path.exists(output_path):
        print(f"Cache valid — skipping stability filtering ({output_path})")
        return output_path

    df = pd.read_csv(input_path, index_col="material_id")
    print(f"Loaded {len(df)} materials from {input_path}")

    if "energy_above_hull" not in df.columns:
        hull_df = pd.read_csv(hull_source_path, index_col="material_id")
        if "energy_above_hull" not in hull_df.columns:
            raise ValueError(
                f"Neither {input_path} nor {hull_source_path} has an energy_above_hull "
                "column -- run data.retrieval.retrieve_data() first to fetch it."
            )
        df = df.join(hull_df[["energy_above_hull"]], how="left")
        print(f"Backfilled energy_above_hull from {hull_source_path} ({input_path} predates that column)")

    df = df.dropna(subset=REQUIRED_COLUMNS + ["energy_above_hull"])
    print(f"{len(df)} materials remaining after dropping nulls")

    stable = df[df["energy_above_hull"] <= max_e_above_hull]
    print(
        f"Stability filter: kept {len(stable)} / {len(df)} materials with "
        f"energy_above_hull <= {max_e_above_hull} eV/atom"
    )

    deduped = _dedupe_by_composition(stable, prefer_min_col="energy_above_hull")
    print(
        f"{len(deduped)} materials retained after keeping only the lowest-energy_above_hull "
        f"structure per composition ({len(stable)} before dedup)"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    deduped.to_csv(output_path, index=True)
    write_meta(output_path, cache_inputs, settings, extra_files=(_PREPROCESSING_PY,))
    print(f"Stability-filtered dataset saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=RAW_DATA_PATH)
    parser.add_argument("--threshold", type=float, default=ELMD_DISTANCE_THRESHOLD)
    parser.add_argument("--similarity", default="mod_petti")
    parser.add_argument("--np", type=int, default=10, dest="n_processes")
    args = parser.parse_args()

    path = preprocess(args.input, args.threshold, args.similarity, args.n_processes)
    print(f"Filtered data at: {path}")
