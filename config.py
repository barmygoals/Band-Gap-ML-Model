"""Project-wide constants: dataset paths, the MP fields retrieved, and the filtering thresholds behind the pinned corpus."""
import os

DATASET_FOLDER = "datasets"
DIAGRAMS_FOLDER = "diagrams"

RAW_DATA_PATH = os.path.join(DATASET_FOLDER, "mp_data.csv")
EXPT_GAP_PATH = os.path.join(DATASET_FOLDER, "expt_gap.csv")

API_PROPERTIES = [
    "formula_pretty",      # chemical formula (e.g. H2O)
    "band_gap",            # band gap in eV
    "energy_above_hull",   # DFT stability in eV/atom (see MAX_ENERGY_ABOVE_HULL)
]
REQUIRED_COLUMNS = ["formula_pretty", "band_gap"]   # a material lacking either is dropped

# Taken from the nested `symmetry` field, which retrieve_data fetches separately.
SYMMETRY_COLUMNS = ["crystal_system", "spacegroup_symbol"]

# ElMD distance below which two compositions count as redundant.
# Higher values filter harder, the opposite of how a distance threshold reads.
ELMD_DISTANCE_THRESHOLD = 5.0

# The standard DFT screening cutoff for synthesisability, 150 meV/atom.
# Regeneration path only, as PINNED_DATASET was built at hull <= 0 (METHODOLOGY.md §4).
MAX_ENERGY_ABOVE_HULL = 0.15

# Drop a composition holding an element in fewer than this many compositions of that corpus.
# The smallest cutoff the pinned corpus can anchor every element at (DECISIONS.md 2026-08-15).
# Callers pass it explicitly, as preprocessing.py's md5 keys every MD-HIT dataset.
MIN_ELEMENT_COUNT = 6

# The pinned dataset every module defaults to, md5 d60d2cb6ce286b8da0a134ef150bb703.
# Rebuild it with scripts/build_element_filtered.py.
# Not 94ff3b469f09.csv, same content but unreachable now (DECISIONS.md 2026-08-15).
PINNED_DATASET = os.path.join('cache', 'element_filtered', '5195089b6296.csv')

# True derives the fraction vocabulary from the dataset, False uses the fixed 118-element table.
# True, else _fit_pipeline decided per fold and folds of one suite differed in feature space.
# A constant rather than a parameter, as it is part of the nn_features cache key.
DERIVE_ELEMENT_VOCABULARY = True

# The seeds every multi-seed arm uses (42 is also the canonical split and export seed).
# Master-plan YAMLs spell their seeds out, so this covers code-level defaults alone.
TRAINING_SEEDS = [42, 43, 44]
