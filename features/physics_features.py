"""Physics-motivated feature tier for the input-information ladder, pre-registered
2026-07-16, with five groups each named to a mechanism. Computed per composition and
cache-free, on top of build_nn_features' standard tier.

  parity      : total valence electron count per formula unit is odd -> cannot be
                a simple band insulator (metallicity prior).
  vec_octet   : valence electrons per atom + distance to 8- and 18-electron counts
                (the magic-number family).
  ionicity    : max pairwise Pauling electronegativity difference + mean Mulliken
                electronegativity (IE+EA)/2 (band-edge estimator).
  screening   : stoichiometric mean atomic polarizability (dielectric proxy).
  relativity  : f-electron fraction + mean atomic number (SOC / +U flags).
"""
import numpy as np
from pymatgen.core import Composition, Element

# Static dipole polarizabilities (atomic units), Schwerdtfeger & Nagle 2019 recommended values.
# DOI 10.1080/00268976.2018.1535143.
# Transcribed from memory and needing checking against the paper before any figure quotes one.
# In this file, so the physics_descriptors fingerprint sees a correction.
_POLARIZABILITY_AU = {
    'H': 4.51, 'He': 1.38, 'Li': 164.1, 'Be': 37.7, 'B': 20.5, 'C': 11.3,
    'N': 7.4, 'O': 5.3, 'F': 3.74, 'Ne': 2.66, 'Na': 162.7, 'Mg': 71.2,
    'Al': 57.8, 'Si': 37.3, 'P': 25.0, 'S': 19.4, 'Cl': 14.6, 'Ar': 11.08,
    'K': 289.7, 'Ca': 160.8, 'Sc': 97.0, 'Ti': 100.0, 'V': 87.0, 'Cr': 83.0,
    'Mn': 68.0, 'Fe': 62.0, 'Co': 55.0, 'Ni': 49.0, 'Cu': 46.5, 'Zn': 38.7,
    'Ga': 50.0, 'Ge': 40.0, 'As': 30.0, 'Se': 28.9, 'Br': 21.0, 'Kr': 16.78,
    'Rb': 319.8, 'Sr': 197.2, 'Y': 162.0, 'Zr': 112.0, 'Nb': 98.0, 'Mo': 87.0,
    'Tc': 79.0, 'Ru': 72.0, 'Rh': 66.0, 'Pd': 26.1, 'Ag': 55.0, 'Cd': 46.0,
    'In': 65.0, 'Sn': 53.0, 'Sb': 43.0, 'Te': 38.0, 'I': 32.9, 'Xe': 27.32,
    'Cs': 400.9, 'Ba': 272.0, 'La': 215.0, 'Ce': 205.0, 'Pr': 216.0, 'Nd': 208.0,
    'Pm': 200.0, 'Sm': 192.0, 'Eu': 184.0, 'Gd': 158.0, 'Tb': 170.0, 'Dy': 163.0,
    'Ho': 156.0, 'Er': 150.0, 'Tm': 144.0, 'Yb': 139.0, 'Lu': 137.0, 'Hf': 103.0,
    'Ta': 74.0, 'W': 68.0, 'Re': 62.0, 'Os': 57.0, 'Ir': 54.0, 'Pt': 48.0,
    'Au': 36.0, 'Hg': 33.9, 'Tl': 50.0, 'Pb': 47.0, 'Bi': 48.0, 'Th': 217.0,
    'U': 129.0,
}

def physics_features(comp: Composition) -> dict:
    els, amts = zip(*[(e, float(a)) for e, a in comp.items()])
    fr = np.array(amts) / sum(amts)
    def _safe(fn, el, default=np.nan):
        try:
            v = fn(el); return float(v) if v is not None else default
        except Exception:
            return default
    # valence count via Element.valence (group-based s+p+d estimate) when available
    vec = []
    for el in els:
        try:
            v = el.valence; vec.append(float(v[1]) if v else np.nan)
        except Exception:
            vec.append(np.nan)
    vec = np.array(vec); tot_val = np.nansum(vec * np.array(amts))
    chi = np.array([_safe(lambda e: e.X, el) for el in els])
    ie  = np.array([_safe(lambda e: e.ionization_energies[0] if e.ionization_energies else None, el) for el in els])
    ea  = np.array([_safe(lambda e: e.electron_affinity, el) for el in els])
    pol = np.array([_POLARIZABILITY_AU.get(el.symbol, np.nan) for el in els])
    z   = np.array([el.Z for el in els], dtype=float)
    f_el = np.array([1.0 if el.is_lanthanoid or el.is_actinoid else 0.0 for el in els])
    vpa = float(np.nansum(fr * vec))
    return {
        'phys_parity_odd': float(round(tot_val) % 2) if not np.isnan(tot_val) else np.nan,
        'phys_vec_per_atom': vpa,
        'phys_dist_octet': abs(vpa - 8.0), 'phys_dist_18e': abs(vpa - 18.0),
        'phys_max_dchi': float(np.nanmax(chi) - np.nanmin(chi)) if len(els) > 1 else 0.0,
        'phys_mulliken_chi': float(np.nansum(fr * (ie + ea) / 2.0)),
        # Plain sum rather than nansum, so one missing element NaNs the compound for imputation.
        'phys_mean_polarizability': float(np.sum(fr * pol)),
        'phys_f_fraction': float(np.nansum(fr * f_el)),
        'phys_mean_Z': float(np.nansum(fr * z)),
    }
