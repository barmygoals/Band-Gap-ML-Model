"""Phase-0 gate asserting the frozen input datasets are byte-identical to the ones every
documented number came from, and failing the chain on any drift. Re-fetching from the MP API
is deliberately not part of the plan, as MP evolves and the pinned CSV does not.

Two modes, dispatched by main() on which params the plan step provides.
  file mode  (data_path, expected_md5, expected_rows) covers one CSV.
  tree mode  (dir_path, expected_files, expected_manifest_md5) covers a directory, as an md5
             over the sorted "name:md5" manifest. *.csv only, as the .meta.json sidecars
             carry timestamps and are provenance rather than input.
"""
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def verify_dataset(data_path: str, expected_md5: str, expected_rows: int) -> dict:
    md5 = _file_md5(data_path)
    with open(data_path, encoding="utf-8") as f:
        rows = sum(1 for _ in f) - 1
    if md5 != expected_md5 or rows != expected_rows:
        raise RuntimeError(
            f"DATASET DRIFT: {data_path} has md5={md5} rows={rows}, expected "
            f"md5={expected_md5} rows={expected_rows}. The master plan only proves "
            "anything on the frozen inputs -- do not proceed; re-pin deliberately if "
            "the change is intentional (and log it in DECISIONS.md)."
        )
    print(f"verified {data_path}: md5={md5} rows={rows}")
    return {"data_path": data_path, "md5": md5, "rows": rows}


def verify_tree(dir_path: str, expected_files: int,
                expected_manifest_md5: str | None = None,
                allow_absent: bool = False) -> dict:
    # Absent-but-expected (2026-08-15), as phase 0's preflight runs before the prebuild job.
    # Only a wholly absent directory passes, since a half-built one is what a gate must catch.
    if allow_absent and not os.path.isdir(dir_path):
        print(f"ABSENT, allowed: {dir_path} is not built yet -- it is produced by the "
              "prebuild job and this gate is re-checked in-chain, after that job succeeds.")
        return {"dir_path": dir_path, "files": 0, "absent": True}

    files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    manifest = "".join(f"{os.path.basename(p)}:{_file_md5(p)}\n" for p in files)
    manifest_md5 = hashlib.md5(manifest.encode()).hexdigest()

    # Record mode (2026-08-15) for an artifact this campaign builds, whose hash is unpinnable.
    # The file count is still asserted, so a build killed part-way still fails.
    if expected_manifest_md5 is None:
        if len(files) != expected_files:
            raise RuntimeError(
                f"INCOMPLETE ARTIFACT: {dir_path} has {len(files)} CSVs, expected "
                f"{expected_files}. A partially-built artifact would silently give some "
                "folds no subset to train on -- do not proceed; re-run the build."
            )
        print(f"RECORDED (unpinned) {dir_path}: {len(files)} CSVs, "
              f"manifest md5={manifest_md5}")
        print(f"  -> pin this in phase0_verify.yaml: expected_manifest_md5: {manifest_md5}")
        return {"dir_path": dir_path, "files": len(files),
                "manifest_md5": manifest_md5, "pinned": False}

    if len(files) != expected_files or manifest_md5 != expected_manifest_md5:
        raise RuntimeError(
            f"FOLD-SET DRIFT: {dir_path} has {len(files)} CSVs, manifest md5="
            f"{manifest_md5}; expected {expected_files} files, md5="
            f"{expected_manifest_md5}. A missing, added or altered fold file makes "
            "every fold-based comparison non-comparable -- do not proceed; re-pin "
            "deliberately if intentional (and log it in DECISIONS.md)."
        )
    print(f"verified {dir_path}: {len(files)} CSVs, manifest md5={manifest_md5}")
    return {"dir_path": dir_path, "files": len(files), "manifest_md5": manifest_md5}


def main(**params) -> dict:  # plan-step entry point (run_experiments._script_step)
    if "dir_path" in params:
        return verify_tree(**params)
    return verify_dataset(**params)
