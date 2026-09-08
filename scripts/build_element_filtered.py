"""Build the rare-element-filtered corpora at config.MIN_ELEMENT_COUNT.

Every corpus is filtered as the last step of its own construction. Counting per corpus
matters, as rarity is not inherited downwards and filtering the parent alone leaves the rows
that made the 2026-08-15 campaign diverge.

In scripts/ and never editing data/preprocessing.py, whose md5 keys every MD-HIT dataset.

Prints the path, md5 and row count of each artifact, which is what phase 0's gates read.

Run: uv run python scripts/build_element_filtered.py [--check]
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from cache import file_md5
from config import MIN_ELEMENT_COUNT, RAW_DATA_PATH
from data.preprocessing import filter_rare_elements

# Both populations, as OOD and transfer run across them and must agree on the element space.
CORPORA: list[tuple[str, str]] = [
    ("pinned (hull<=0)", os.path.join("cache", "stability_filtered", "f203171b1ed2.csv")),
    ("raw MP",           RAW_DATA_PATH),
]


def _report(label: str, path: str) -> dict:
    df = pd.read_csv(path)
    elements = sorted({e for f in df["formula_pretty"].dropna()
                       for e in re.findall(r"[A-Z][a-z]?", str(f))})
    row = {"corpus": label, "path": path, "rows": len(df),
           "elements": len(elements), "md5": file_md5(path)}
    print(f"\n{label}")
    print(f"  path     {path}")
    print(f"  rows     {len(df):,}")
    print(f"  elements {len(elements)}")
    print(f"  md5      {row['md5']}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report existing artifacts only; fail if one is missing")
    args = parser.parse_args()

    print(f"MIN_ELEMENT_COUNT = {MIN_ELEMENT_COUNT} (config.py)")
    rows = []
    for label, src in CORPORA:
        if not os.path.exists(src):
            raise SystemExit(f"missing input for {label}: {src}")
        if args.check:
            from cache import get_cache_path
            from data.preprocessing import _PREPROCESSING_PY
            settings = {"min_element_count": MIN_ELEMENT_COUNT, "formula_column": "formula_pretty"}
            out = get_cache_path("element_filtered", [src], settings,
                                 extra_files=(_PREPROCESSING_PY,))
            if not os.path.exists(out):
                raise SystemExit(f"NOT BUILT at MIN_ELEMENT_COUNT={MIN_ELEMENT_COUNT}: {label} -> {out}")
        else:
            out = filter_rare_elements(src, min_element_count=MIN_ELEMENT_COUNT)
        rows.append(_report(label, out))

    print("\nPin these in config.PINNED_DATASET and experiments/master_v2/phase0_verify.yaml:")
    for r in rows:
        print(f"  {r['corpus']:18s} {r['path']}  md5={r['md5']}  rows={r['rows']}")


if __name__ == "__main__":
    main()
