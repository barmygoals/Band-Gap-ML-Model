"""Per-dataset element statistics and the cross-dataset fidelity and vocabulary checks, matched by parsed composition since mp_data.csv's material_id values are not real MP IDs."""

import json
import os
from collections import Counter

import pandas as pd
from pymatgen.core import Composition, Element

from config import EXPT_GAP_PATH, RAW_DATA_PATH


def is_real_composition(comp: Composition) -> bool:
    """True where every species in comp is a genuine Element rather than a Species or
    DummySpecies, as literature-sourced formula strings parse into a fictional element rather
    than raising and a try/except around Composition() does not catch them."""
    return all(isinstance(el, Element) for el in comp.elements)


def count_elements(compositions: list[Composition]) -> Counter:
    """How many compositions contain each element, counted per material rather than per atom."""
    counts: Counter = Counter()
    for comp in compositions:
        for el in comp.elements:
            counts[el.symbol] += 1
    return counts


def element_summary(element_counts: Counter) -> pd.DataFrame:
    """Build an (element, atomic_number, material_count) frame sorted by atomic number, and print a by-count summary."""
    df = pd.DataFrame(
        [(sym, Element(sym).Z, cnt) for sym, cnt in element_counts.items()],
        columns=["element", "atomic_number", "material_count"],
    ).sort_values("atomic_number")
    print("\n--- Materials per Element (top 20) ---")
    print(df.sort_values("material_count", ascending=False).head(20).to_string(index=False))  # re-sorted by count for this printout alone
    return df


def _reduced_formula(formula: str) -> str | None:
    """Parse a formula to its reduced-composition key, returning None rather than raising on unparseable input."""
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None


def check_composition_fidelity(
    expt_path: str = EXPT_GAP_PATH,
    expt_formula_col: str = "formula_pretty",
    expt_gap_col: str = "band_gap",
    other_path: str = RAW_DATA_PATH,
    other_formula_col: str = "formula_pretty",
    other_gap_col: str = "band_gap",
    compare_band_gaps: bool = True,
) -> dict:
    """How many materials in expt_path also appear in other_path, matched on parsed reduced
    composition rather than any identifier column, returning a dict of summary stats and never
    raising on disagreement. compare_band_gaps additionally reports the mean signed difference,
    MAE and Pearson r over matched compositions, and a mean DFT-experiment offset is expected
    rather than a fidelity failure, so Pearson r is the more informative figure."""
    expt = pd.read_csv(expt_path)
    other = pd.read_csv(other_path)

    expt = expt.assign(_composition=expt[expt_formula_col].apply(_reduced_formula)).dropna(subset=["_composition"])
    other = other.assign(_composition=other[other_formula_col].apply(_reduced_formula)).dropna(subset=["_composition"])

    n_expt = len(expt)
    n_other = len(other)
    matched_compositions = set(expt["_composition"]) & set(other["_composition"])
    n_matched = len(matched_compositions)

    result = {
        "n_expt": n_expt,
        "n_other": n_other,
        "n_matched_compositions": n_matched,
        "pct_matched_compositions": n_matched / n_expt * 100 if n_expt else 0.0,
    }

    if compare_band_gaps:
        # Renamed first, as both sides default to band_gap and the merge would suffix them unreported.
        other_gap_renamed = other[["_composition", other_gap_col]].rename(columns={other_gap_col: "_other_gap"})
        merged = expt.merge(other_gap_renamed, on="_composition", how="inner")
        diff = merged[expt_gap_col] - merged["_other_gap"]
        result.update({
            "n_compared_pairs": len(merged),
            "mean_signed_diff_eV": float(diff.mean()) if len(diff) else float("nan"),
            "mae_eV": float(diff.abs().mean()) if len(diff) else float("nan"),
            "pearson_r": float(merged[expt_gap_col].corr(merged["_other_gap"])) if len(merged) > 1 else float("nan"),
        })
        # The paired vectors behind those scalars, so the offset can be replotted without re-merging.
        result["_pairs"] = pd.DataFrame({
            "composition": merged["_composition"],
            "expt_gap": merged[expt_gap_col],
            "other_gap": merged["_other_gap"],
            "diff_eV": diff,
        })

    summary = (
        f"composition fidelity: {expt_path} vs {other_path}:\n"
        f"  {result['n_matched_compositions']}/{result['n_expt']} expt compositions matched "
        f"({result['pct_matched_compositions']:.1f}%)"
    )
    if compare_band_gaps:
        summary += (
            f"\n  band gap ({expt_gap_col} - {other_gap_col}) over {result['n_compared_pairs']} "
            f"matched pairs (all matching polymorphs, not deduplicated): "
            f"mean={result['mean_signed_diff_eV']:.3f} eV  MAE={result['mae_eV']:.3f} eV  "
            f"Pearson r={result['pearson_r']:.3f}"
        )
    print(summary)
    return result


def _discover_filtered_variants(filtered_dir: str) -> list[tuple[str, str]]:
    """Scan filtered_dir's sidecars for their MD-HIT threshold, returning [(label, csv_path), ...] by ascending threshold."""
    variants = []
    if not os.path.isdir(filtered_dir):
        return variants
    for fname in sorted(os.listdir(filtered_dir)):
        if not fname.endswith(".meta.json"):
            continue
        with open(os.path.join(filtered_dir, fname)) as f:
            settings = json.load(f)["settings"]
        csv_path = os.path.join(filtered_dir, fname[: -len(".meta.json")] + ".csv")
        variants.append((settings.get("threshold"), csv_path))
    variants.sort(key=lambda v: v[0])
    return [(f"threshold={t}", path) for t, path in variants]


def _print_fidelity_table(df: pd.DataFrame) -> None:
    """Print a fixed-width, right-aligned table of compare_expt_vs_dft_variants()'s results."""
    # n_expt and path excluded, one constant across rows and the other in the variant label.
    headers = {
        "variant": "variant", "n_other": "n_other",
        "n_matched_compositions": "matched_comp", "pct_matched_compositions": "pct_matched",
        "n_compared_pairs": "n_pairs", "mean_signed_diff_eV": "mean_diff_eV",
        "mae_eV": "MAE_eV", "pearson_r": "pearson_r",
    }
    display_cols = [c for c in df.columns if c in headers]
    def fmt(col: str, v) -> str:
        if col == "pct_matched_compositions":
            return f"{v:.1f}%"
        if col in ("mean_signed_diff_eV", "mae_eV", "pearson_r"):
            return f"{v:.3f}"
        return str(v)

    str_cols = {c: df[c].apply(lambda v, c=c: fmt(c, v)) for c in display_cols}
    widths = {c: max(len(headers[c]), str_cols[c].str.len().max()) for c in display_cols}

    header = "  ".join(headers[c].rjust(widths[c]) if c != "variant" else headers[c].ljust(widths[c]) for c in display_cols)
    print(header)
    print("-" * len(header))
    for i in range(len(df)):
        row = "  ".join(
            str_cols[c].iloc[i].rjust(widths[c]) if c != "variant" else str_cols[c].iloc[i].ljust(widths[c])
            for c in display_cols
        )
        print(row)


def compare_expt_vs_dft_variants(
    expt_path: str = EXPT_GAP_PATH,
    raw_path: str = RAW_DATA_PATH,
    filtered_dir: str = "cache/filtered_data",
    compare_band_gaps: bool = True,
    save_path: str | None = None,
    print_table: bool = True,
) -> pd.DataFrame:
    """check_composition_fidelity() between expt_path and both raw_path and every
    MD-HIT-filtered variant under filtered_dir, one row per variant, discovered from their
    .meta.json sidecars and optionally written to save_path."""
    variants = [("raw", raw_path)] + _discover_filtered_variants(filtered_dir)

    rows = []
    pair_frames = []
    for label, path in variants:
        result = check_composition_fidelity(expt_path=expt_path, other_path=path, compare_band_gaps=compare_band_gaps)
        pairs = result.pop("_pairs", None)  # out of the summary row, written separately
        if pairs is not None:
            pair_frames.append(pairs.assign(variant=label))
        rows.append({"variant": label, "path": path, **result})
    df = pd.DataFrame(rows)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\nSaved comparison table to {save_path}")
        if pair_frames:
            pairs_path = save_path.replace(".csv", "_pairs.csv")
            pd.concat(pair_frames, ignore_index=True).to_csv(pairs_path, index=False)
            print(f"Saved per-composition gap pairs to {pairs_path}")

    if print_table:
        print()
        _print_fidelity_table(df)

    return df


def _elements_in(path: str, formula_col: str) -> set[str]:
    """Element symbols anywhere in path's formula_col, skipping unparseable formulas with the same tolerance as model.residual.prepare_expt_features."""
    df = pd.read_csv(path).dropna(subset=[formula_col])
    elements: set[str] = set()
    for f in df[formula_col]:
        try:
            comp = Composition(f)
        except Exception:
            continue
        if not is_real_composition(comp):
            continue
        elements.update(el.symbol for el in comp.elements)
    return elements


def compare_element_vocabularies(
    path_a: str,
    path_b: str,
    formula_col_a: str = "formula_pretty",
    formula_col_b: str = "formula_pretty",
) -> dict:
    """Compare the element sets of two datasets, rather than the whole-composition overlap
    check_composition_fidelity gives, returning n_elements_a, n_elements_b, n_shared, only_in_a
    and only_in_b (by atomic number), a_subset_of_b and b_subset_of_a. Since 2026-08-15 the
    fraction columns are the corpus's own vocabulary, so an element present only in b has no
    column in a's matrix and a caller crossing populations has to reindex by name first, as
    model.residual does, making b_subset_of_a the direction in which a's model applies to b."""
    elements_a = _elements_in(path_a, formula_col_a)
    elements_b = _elements_in(path_b, formula_col_b)
    only_a = sorted(elements_a - elements_b, key=lambda s: Element(s).Z)
    only_b = sorted(elements_b - elements_a, key=lambda s: Element(s).Z)

    result = {
        "n_elements_a": len(elements_a),
        "n_elements_b": len(elements_b),
        "n_shared": len(elements_a & elements_b),
        "only_in_a": only_a,
        "only_in_b": only_b,
        "a_subset_of_b": not only_a,
        "b_subset_of_a": not only_b,
    }
    print(
        f"element vocabulary: {path_a} ({result['n_elements_a']} elements) vs "
        f"{path_b} ({result['n_elements_b']} elements) -- {result['n_shared']} shared, "
        f"{len(only_a)} only in a, {len(only_b)} only in b"
    )
    if only_a:
        print(f"  only in {path_a}: {', '.join(only_a)}")
    if only_b:
        print(f"  only in {path_b}: {', '.join(only_b)}")
    if result["a_subset_of_b"] and result["b_subset_of_a"]:
        print("  subset check: vocabularies are identical")
    elif result["a_subset_of_b"]:
        print("  subset check: a's vocabulary is a subset of b's")
    elif result["b_subset_of_a"]:
        print("  subset check: b's vocabulary is a subset of a's -- a model trained on a has seen every element in b")
    else:
        print("  subset check: neither vocabulary contains the other")
    return result
