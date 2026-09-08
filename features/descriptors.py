"""Magpie-style compositional descriptors, matminer ElementProperty descriptors and element fraction vectors for pymatgen Compositions, cached to CSV."""
import math
import os
from typing import cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matminer.featurizers.composition import ElementProperty, Meredig
from matminer.featurizers.composition.orbital import AtomicOrbitals
from matminer.utils.data import MagpieData
from pymatgen.core import Composition, DummySpecies, Element, Species

from cache import get_cache_path, write_meta

_ELEMENT_STATS_FEATURES = [
    "Number", "Row", "Column",
    "NsValence", "NpValence", "NdValence", "NfValence", "NValence",
    "NsUnfilled", "NpUnfilled", "NdUnfilled", "NfUnfilled", "NUnfilled",
]


def _get_valence_electrons(el: Element | Species | DummySpecies) -> float | None:
    """Valence electron count for an element."""
    try:
        v = el.valence
        return float(v[1]) if v is not None else None
    except ValueError:
        return None  # ambiguous valence, excluded from the average rather than treated as 0


def _finite_or_none(x) -> float | None:
    """Normalise a pymatgen property to "a real number, or None", as pymatgen signals a missing
    value inconsistently and this module's filters test `v is not None`, so a NaN would slip
    through and corrupt the stats. min and max also become order-dependent, as Python
    propagates NaN positionally, so every getter has to funnel through here."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _get_electronegativity(el: Element) -> float | None:
    """Pauling electronegativity (NaN for noble gases)."""
    return _finite_or_none(el.X)


def _get_atomic_radius(el: Element) -> float | None:
    """Atomic radius, falling back to the calculated value where empirical is absent."""
    r = el.atomic_radius if el.atomic_radius is not None else el.atomic_radius_calculated
    return _finite_or_none(r)


def _get_atomic_mass(el: Element) -> float:
    """Atomic mass."""
    return float(el.atomic_mass)


def _get_ionisation_energy(el: Element) -> float | None:
    """First ionisation energy, eV."""
    try:
        ies = el.ionization_energies  # eV, 0-indexed, so [0] is the first IE
        return _finite_or_none(ies[0]) if ies else None
    except (AttributeError, IndexError, TypeError):
        return None


_PROP_GETTERS = {   # property name -> getter
    "valence_electrons": _get_valence_electrons,
    "electronegativity": _get_electronegativity,
    "atomic_radius": _get_atomic_radius,
    "atomic_mass": _get_atomic_mass,
    "ionization_energy": _get_ionisation_energy,
}


def _stoich_weighted_mean(comp: Composition, value_cache: dict[str, float | None]) -> float | None:
    """Stoichiometry-weighted mean of a property from a pre-built element cache, None where every element lacks it."""
    pairs = [(value_cache.get(el.symbol), float(amt)) for el, amt in comp.items()]
    pairs = [(v, w) for v, w in pairs if v is not None]
    if not pairs:
        return None
    values, weights = zip(*pairs)
    total = sum(weights)
    return sum((w / total) * v for w, v in zip(weights, values))


# Magpie-style descriptors computed here through pymatgen rather than matminer.

def _featurise_comp(comp: Composition, prop_cache: dict) -> dict[str, float | None]:
    """Magpie-style descriptors for one composition, each property yielding mean, min, max and range."""
    row: dict[str, float | None] = {}
    for prop, cache in prop_cache.items():
        pairs = [(cache.get(el.symbol), float(amt)) for el, amt in comp.items()]
        pairs = [(v, w) for v, w in pairs if v is not None]
        if not pairs:
            for stat in ("mean", "min", "max", "range"):
                row[f"{prop}_{stat}"] = None
            continue
        values, weights = zip(*pairs)
        total = sum(weights)
        fracs = [w / total for w in weights]
        wmean = sum(f * v for f, v in zip(fracs, values))
        vmin, vmax = min(values), max(values)  # bare elemental extremes, the standard Magpie convention
        row[f"{prop}_mean"] = wmean
        row[f"{prop}_min"] = vmin
        row[f"{prop}_max"] = vmax
        row[f"{prop}_range"] = vmax - vmin
    return row


def compute_magpie_descriptors(compositions: list[Composition], data_path: str) -> pd.DataFrame:
    """Magpie-style compositional descriptors, one row per composition (cached)."""
    settings = {"properties": list(_PROP_GETTERS.keys()), "stats": ["mean", "min", "max", "range"]}
    cache = get_cache_path("magpie_descriptors", [data_path], settings)
    if os.path.exists(cache):
        print(f"\nLoading cached Magpie descriptors from {cache}")
        return pd.read_csv(cache)

    all_symbols = {el.symbol for comp in compositions for el in comp.elements}
    # one pymatgen attribute lookup per unique element rather than per composition
    prop_cache = {
        prop: {sym: getter(Element(sym)) for sym in all_symbols}
        for prop, getter in _PROP_GETTERS.items()
    }

    raw = Parallel(n_jobs=-1)(  # joblib rather than matminer's Pool, which mis-parallelises
        delayed(_featurise_comp)(comp, prop_cache) for comp in compositions
    )
    records = cast(list[dict[str, float | None]], raw)

    desc_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    desc_df.to_csv(cache, index=False)
    write_meta(cache, [data_path], settings)
    print(f"\nMagpie descriptors saved to {cache}")
    print("\n--- Magpie Descriptors Summary ---")
    print(desc_df.describe().to_string())
    return desc_df


def compute_element_stats(compositions: list[Composition], data_path: str) -> pd.DataFrame:
    """matminer ElementProperty (MagpieData) descriptors, one row per composition (cached)."""
    settings = {"data_source": "magpie", "features": _ELEMENT_STATS_FEATURES, "stats": ["mean"]}
    cache = get_cache_path("element_stats", [data_path], settings)
    if os.path.exists(cache):
        print(f"\nLoading cached ElementProperty features from {cache}")
        return pd.read_csv(cache)

    magpie = MagpieData()
    all_symbols = {el.symbol for comp in compositions for el in comp.elements}
    feature_cache = {
        feat: {sym: magpie.get_elemental_property(Element(sym), feat) for sym in all_symbols}
        for feat in _ELEMENT_STATS_FEATURES
    }
    col_names = [f"MagpieData mean {feat}" for feat in _ELEMENT_STATS_FEATURES]  # matminer's naming

    records = []
    for comp in compositions:
        row: dict[str, float | None] = {}
        for feat, col in zip(_ELEMENT_STATS_FEATURES, col_names):
            row[col] = _stoich_weighted_mean(comp, feature_cache[feat])
        records.append(row)

    result = pd.DataFrame(records)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    result.to_csv(cache, index=False)
    write_meta(cache, [data_path], settings)
    print(f"\nElementProperty features saved to {cache}")
    print("\n--- ElementProperty (MagpieData) Summary ---")
    print(result.describe().to_string())
    return result


# Columns from each matminer preset below, matching Jung, Jung & Cole (2024)'s BGML repo.
# total_e is Composition.total_electrons rather than a valence-electron count.
_MAGPIE_COLS = ["MagpieData avg_dev Electronegativity", "MagpieData avg_dev NpValence", "MagpieData avg_dev GSbandgap"]
_MATMINER_COLS = ["PymatgenData std_dev group", "PymatgenData mean block"]
_DEML_COLS = ["DemlData maximum electronegativity"]
_MEREDIG_COLS = ["frac p valence electrons", "frac d valence electrons"]


def _featurise_literature(
    comp: Composition,
    magpie: ElementProperty,
    matminer_fz: ElementProperty,
    deml: ElementProperty,
    meredig: Meredig,
    orbitals: AtomicOrbitals,
) -> dict[str, float | None]:
    """One row of compute_literature_descriptors(), via matminer's presets rather than reimplementing their statistics."""
    row: dict[str, float | None] = {}
    try:
        magpie_vals = dict(zip(magpie.feature_labels(), magpie.featurize(comp)))
        for col in _MAGPIE_COLS:
            row[col] = magpie_vals[col]

        matminer_vals = dict(zip(matminer_fz.feature_labels(), matminer_fz.featurize(comp)))
        for col in _MATMINER_COLS:
            row[col] = matminer_vals[col]

        deml_vals = dict(zip(deml.feature_labels(), deml.featurize(comp)))
        row["DemlData maximum electronegativity"] = deml_vals["DemlData maximum electronegativity"]

        meredig_vals = dict(zip(meredig.feature_labels(), meredig.featurize(comp)))
        row["frac_p_valence_electrons"] = meredig_vals["frac p valence electrons"]
        row["frac_d_valence_electrons"] = meredig_vals["frac d valence electrons"]

        row["total_e"] = comp.total_electrons

        orbital_vals = dict(zip(orbitals.feature_labels(), orbitals.featurize(comp)))
        row["HOMO_energy"] = orbital_vals["HOMO_energy"]
    except Exception:
        for col in (*_MAGPIE_COLS, *_MATMINER_COLS, *_DEML_COLS,
                    "frac_p_valence_electrons", "frac_d_valence_electrons", "total_e", "HOMO_energy"):
            row.setdefault(col, None)
    return row


def compute_literature_descriptors(compositions: list[Composition], data_path: str) -> pd.DataFrame:
    """The 10 interpretable features of Jung, Jung & Cole (2024)'s top-20 band-gap predictors,
    through matminer's presets so the values match their BGML repo exactly. Their other 10 are
    MEGNet embeddings, needing pretrained weights this project does not use."""
    settings = {"version": 2}
    cache = get_cache_path("literature_descriptors", [data_path], settings)
    if os.path.exists(cache):
        print(f"\nLoading cached literature descriptors from {cache}")
        return pd.read_csv(cache)

    magpie = ElementProperty.from_preset("magpie")
    matminer_fz = ElementProperty.from_preset("matminer")
    deml = ElementProperty.from_preset("deml")
    meredig = Meredig()
    orbitals = AtomicOrbitals()

    raw = Parallel(n_jobs=-1)(
        delayed(_featurise_literature)(comp, magpie, matminer_fz, deml, meredig, orbitals)
        for comp in compositions
    )
    records = cast(list[dict[str, float | None]], raw)

    result = pd.DataFrame(records)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    result.to_csv(cache, index=False)
    write_meta(cache, [data_path], settings)
    print(f"\nLiterature descriptors saved to {cache}")
    print("\n--- Literature Descriptors Summary ---")
    print(result.describe().to_string())
    return result


# Fixed vocabulary (H..Og), now the fallback for config.DERIVE_ELEMENT_VOCABULARY=False.
# Its concatenability argument was retired, as a name-based reindex is correct across datasets.
_ALL_ELEMENT_SYMBOLS: list[str] = [Element.from_Z(z).symbol for z in range(1, 119)]


def compute_fraction_vectors(compositions: list[Composition], data_path: str,
                             derive_vocabulary: bool | None = None) -> pd.DataFrame:
    """Element fraction vectors, one row per composition and one column per element.
    derive_vocabulary=False uses the fixed 118-element table, preserving existing cache entries,
    whilst True builds the vocabulary from the compositions present (supervisor request,
    2026-08-15), else _fit_pipeline decides which element columns survive per fold. A
    whole-dataset vocabulary is not leakage, as which elements exist is a property of the corpus
    and not of any label, though callers whose train and test rows come from different files
    must pass the union of both."""
    # None follows config.DERIVE_ELEMENT_VOCABULARY, so the policy is set in one place.
    if derive_vocabulary is None:
        from config import DERIVE_ELEMENT_VOCABULARY as derive_vocabulary
    elements = (sorted({el for c in compositions for el in c.get_el_amt_dict()})
                if derive_vocabulary else _ALL_ELEMENT_SYMBOLS)
    settings = {"elements": elements}
    cache = get_cache_path("fraction_vectors", [data_path], settings)
    if os.path.exists(cache):
        print(f"\nLoading cached fraction vectors from {cache}")
        return pd.read_csv(cache)

    records = []
    for comp in compositions:
        frac = comp.fractional_composition.as_dict()
        records.append({el: frac.get(el, 0.0) for el in elements})

    df = pd.DataFrame(records).fillna(0.0)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.to_csv(cache, index=False)
    write_meta(cache, [data_path], settings)
    # len(elements), as _ALL_ELEMENT_SYMBOLS misreports the count under a derived vocabulary.
    print(f"\nFraction vectors saved to {cache} ({len(elements)} elements)")
    return df


def compute_physics_descriptors(compositions: list[Composition], data_path: str) -> pd.DataFrame:
    """The five pre-registered physics groups from features.physics_features, cached per
    composition as the ladder's rung-3 increment, keyed on physics_features.py so editing a
    group busts the cache."""
    phys_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "physics_features.py")
    settings = {"version": 1}
    cache = get_cache_path("physics_descriptors", [data_path], settings, extra_files=(phys_py,))
    if os.path.exists(cache):
        print(f"\nLoading cached physics descriptors from {cache}")
        return pd.read_csv(cache)

    from features.physics_features import physics_features
    df = pd.DataFrame([physics_features(c) for c in compositions])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.to_csv(cache, index=False)
    write_meta(cache, [data_path], settings, extra_files=(phys_py,))
    print(f"\nPhysics descriptors cached to {cache} ({df.shape[1]} columns)")
    return df


def compute_physics_descriptors_v2(compositions: list[Composition], data_path: str) -> pd.DataFrame:
    """The five audit-surviving v2 physics groups from features.physics_features_v2, giving six
    columns as the "physics_v2" tier's increment. Keyed on physics_features_v2.py and on
    physics_features.py, as v2 imports v1's polarizability table and a table correction has to
    bust this cache too (fingerprint gap closed 2026-07-22)."""
    feat_dir = os.path.dirname(os.path.abspath(__file__))
    extra = (os.path.join(feat_dir, "physics_features_v2.py"),
             os.path.join(feat_dir, "physics_features.py"))
    settings = {"version": 1}
    cache = get_cache_path("physics_descriptors_v2", [data_path], settings, extra_files=extra)
    if os.path.exists(cache):
        print(f"\nLoading cached physics v2 descriptors from {cache}")
        return pd.read_csv(cache)

    from features.physics_features_v2 import physics_features_v2
    # Parallelised like compute_magpie_descriptors, a pure speed change outside the fingerprint.
    raw = Parallel(n_jobs=-1)(delayed(physics_features_v2)(c) for c in compositions)
    df = pd.DataFrame(cast(list[dict], raw))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.to_csv(cache, index=False)
    write_meta(cache, [data_path], settings, extra_files=extra)
    print(f"\nPhysics v2 descriptors cached to {cache} ({df.shape[1]} columns)")
    return df


def compute_physics_descriptors_alt(compositions: list[Composition], data_path: str) -> pd.DataFrame:
    """The "physics-alt" tier from features.physics_features_alt, holding v2's six audited
    columns and the audit-surviving band-edge, hardness and multivalence groups. Keyed on all
    three physics_features*.py files, as it calls v2's function which imports v1's polarizability
    table. Renamed from physics_v3 on 2026-08-17 as an alternative rung rather than a later one,
    the cache name moving with it so the old entries stay in place for the bundled campaigns."""
    feat_dir = os.path.dirname(os.path.abspath(__file__))
    extra = tuple(os.path.join(feat_dir, f) for f in
                  ("physics_features_alt.py", "physics_features_v2.py", "physics_features.py"))
    settings = {"version": 1}
    cache = get_cache_path("physics_descriptors_alt", [data_path], settings, extra_files=extra)
    if os.path.exists(cache):
        print(f"\nLoading cached physics-alt descriptors from {cache}")
        return pd.read_csv(cache)

    from features.physics_features_alt import physics_features_alt
    raw = Parallel(n_jobs=-1)(delayed(physics_features_alt)(c) for c in compositions)
    df = pd.DataFrame(cast(list[dict], raw))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.to_csv(cache, index=False)
    write_meta(cache, [data_path], settings, extra_files=extra)
    print(f"\nPhysics-alt descriptors cached to {cache} ({df.shape[1]} columns)")
    return df


def build_nn_features(
    compositions: list[Composition], data_path: str, feature_tier: str = "standard",
    derive_vocabulary: bool | None = None,
) -> pd.DataFrame:
    """Build the NN input feature matrix, mirroring BGML's feature set.

    feature_tier selects a rung of the input-information ladder.
      "fractions"    element fractions alone.
      "literature"   the 10 literature descriptors alone, the identity-subtraction arm.
      "standard"     fractions with the literature descriptors (default).
      "physics"      standard with the 5 pre-registered physics groups.
      "physics_v2"   standard with the 6 audited v2 groups, an alternative rung 3.
      "physics-alt"  standard with the audited alternative tier (rung 3-alt, 2026-07-22).
    """
    if feature_tier not in ("fractions", "literature", "standard", "physics", "physics_v2", "physics-alt"):
        raise ValueError(f"Unknown feature_tier {feature_tier!r}")
    # version=4 marked the switch to the fixed vocabulary and has not moved since.
    # get_cache_path hashes the settings dict alone, so bumping this by hand retires a stale matrix.
    settings = {"version": 4}
    if feature_tier != "standard":  # keep pre-ladder cache keys byte-identical
        settings["feature_tier"] = feature_tier
    # Recorded only when on, so existing matrices keep their hash.
    # It changes the column count, so it has to be in the key.
    if derive_vocabulary is None:
        from config import DERIVE_ELEMENT_VOCABULARY as _dv
        derive_vocabulary = _dv
    if derive_vocabulary:
        settings["derive_vocabulary"] = True
    cache = get_cache_path("nn_features", [data_path], settings)
    if os.path.exists(cache):
        print(f"\nLoading cached NN feature matrix from {cache}")
        return pd.read_csv(cache)

    parts = []
    if feature_tier != "literature":  # every tier except literature-only carries identity
        parts.append(compute_fraction_vectors(
            compositions, data_path, derive_vocabulary=derive_vocabulary).reset_index(drop=True))
    if feature_tier != "fractions":  # every tier except fractions-only carries the 10 lit cols
        parts.append(compute_literature_descriptors(compositions, data_path).reset_index(drop=True))
    if feature_tier == "physics":
        parts.append(compute_physics_descriptors(compositions, data_path).reset_index(drop=True))
    if feature_tier == "physics_v2":
        parts.append(compute_physics_descriptors_v2(compositions, data_path).reset_index(drop=True))
    if feature_tier == "physics-alt":
        parts.append(compute_physics_descriptors_alt(compositions, data_path).reset_index(drop=True))
    combined = pd.concat(parts, axis=1)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    combined.to_csv(cache, index=False)
    write_meta(cache, [data_path], settings)
    print(f"\nNN feature matrix cached to {cache}")
    print(f"  {combined.shape[0]} samples × {combined.shape[1]} features "
          f"(tier={feature_tier}: " + " + ".join(str(p.shape[1]) for p in parts) + ")")
    return combined
