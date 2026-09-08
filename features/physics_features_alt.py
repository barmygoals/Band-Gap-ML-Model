"""Physics feature tier "physics-alt" (2026-07-22, named physics_v3 until 2026-08-17): v2's
six audited columns plus three new groups, giving four columns.

  band_edges    atomic-orbital gap (LUMO - HOMO), unclipped, the band edge the literature
                tier lacks.
  hardness      chemical hardness of the softest atom, bounding the dissociation-limit gap.
  multivalence  share of atoms with two or more common positive oxidation states, and the
                weighted span of those states.

Audit gate and rejected candidates: DECISIONS.md 2026-07-22. Its own module so v1 and v2
cache fingerprints stay untouched.
"""
import numpy as np
from pymatgen.core import Composition

from features.physics_features_v2 import physics_features_v2

# Lazy, so matminer stays an import-time no-op for tiers that skip it.
_AO = None


def _atomic_orbitals():
    global _AO
    if _AO is None:
        from matminer.featurizers.composition import AtomicOrbitals
        _AO = AtomicOrbitals()
    return _AO


def _ao_gap(comp: Composition) -> float:
    """gap_AO (LUMO - HOMO) via the same matminer featurizer the literature tier uses for HOMO_energy."""
    try:
        ao = _atomic_orbitals()
        vals = dict(zip(ao.feature_labels(), ao.featurize(comp)))
        return float(vals["gap_AO"])
    except Exception:
        return np.nan


def physics_features_alt(comp: Composition) -> dict:
    row = physics_features_v2(comp)  # the six audited v2 columns, byte-identical

    els, amts = zip(*[(e, float(a)) for e, a in comp.items()])
    fr = np.array(amts) / sum(amts)

    def _safe(fn, el, default=np.nan):
        try:
            v = fn(el)
            return float(v) if v is not None else default
        except Exception:
            return default

    ie = np.array([_safe(lambda e: e.ionization_energies[0] if e.ionization_energies else None, el) for el in els])
    ea = np.array([_safe(lambda e: e.electron_affinity, el) for el in els])
    hardness = (ie - ea) / 2.0

    multi = np.array([1.0 if len([s for s in el.common_oxidation_states if s > 0]) >= 2 else 0.0
                      for el in els])
    span = np.array([float(max(el.common_oxidation_states) - min(el.common_oxidation_states))
                     if el.common_oxidation_states else 0.0 for el in els])

    # Plain min rather than nanmin, so one missing element NaNs the compound for imputation.
    row.update({
        'phys_ao_gap': _ao_gap(comp),
        'phys_hardness_min': float(np.min(hardness)),
        'phys_frac_multivalent': float(np.sum(fr * multi)),
        'phys_oxstate_span': float(np.sum(fr * span)),
    })
    return row
