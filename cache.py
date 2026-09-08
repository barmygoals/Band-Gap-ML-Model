"""Two caching patterns, content-addressed (get_cache_path) and fixed-path fingerprint
validation (fingerprint_valid and write_meta).

A computation's identity is a fingerprint over each input data file, each `extra_files`
source file and the settings dict, so any change gives a new path and old results are never
overwritten.

Data files hash by raw bytes. Source files go through source_md5(), which since 2026-08-22
hashes their code rather than their bytes, so rewriting a comment does not re-key every
artifact whilst a real code change still invalidates (scripts/build_code_pins.py).

`extra_files` is hand-maintained and nothing verifies it is complete, so a computation that
gains a dependency needs the file listing here.
"""
import ast
import datetime
import functools
import hashlib
import json
import os
import sys

import pandas as pd

CACHE_DIR = "cache"
_PINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_pins.json")
_PIN_NOTICE_SHOWN = False


@functools.lru_cache(maxsize=None)
def file_md5(path: str) -> str:
    """Return the MD5 hash of a file, where the lru_cache lasts the whole process so a
    fixed-path file rewritten mid-run keeps its old hash, which is why plans put retrieval
    first. Chunked reading bounds memory rather than speed."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def _pins() -> dict:
    """Returns code_pins.json's files map, or {} where it is absent or unreadable, absence not
    being an error as without pins every source file hashes by raw bytes."""
    try:
        with open(_PINS_PATH) as f:
            return json.load(f).get("files", {})
    except (OSError, ValueError):
        return {}


@functools.lru_cache(maxsize=None)
def _semantic_digest(path: str) -> str:
    """Returns an MD5 of the file's code, with comments and docstrings removed, as ast.parse
    discards comments and ast.dump omits line numbers so only a change to the code itself moves
    this digest. It has to stay byte-identical to scripts/build_code_pins.semantic_digest(),
    since the pins stop applying if the two copies drift."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return hashlib.md5(ast.dump(tree).encode()).hexdigest()


def source_md5(path: str) -> str:
    """Returns the cache-key digest of an extra_files source file.

    Whilst the code is unchanged this gives the md5 it had when the pinned campaign ran, so
    rewriting comments for publication does not rename a cache entry and trigger a retrain.

    Once real code changes the semantic digest no longer matches its pin and this falls
    through to the true md5. The same fallback covers a missing pin file, an unpinned file, an
    unparseable one and an interpreter the pins were not generated under, each a cache miss
    rather than a wrong hit.
    """
    global _PIN_NOTICE_SHOWN
    pin = _pins().get(os.path.relpath(os.path.abspath(path), os.path.dirname(_PINS_PATH))
                      .replace(os.sep, "/"))
    if not pin:
        return file_md5(path)
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    recorded = pin.get("semantic", {}).get(tag)
    if not recorded:
        return file_md5(path)
    try:
        if _semantic_digest(path) != recorded:
            return file_md5(path)
    except (OSError, SyntaxError, ValueError):
        return file_md5(path)
    if pin["md5"] != file_md5(path) and not _PIN_NOTICE_SHOWN:
        # Reported once per process, as the run's keys now describe code the files no longer match.
        _PIN_NOTICE_SHOWN = True
        print(f"code pins active ({tag}): source files differ from the pinned campaign in "
              f"comments and docstrings only, cache keys preserved. "
              f"Check with scripts/build_code_pins.py --verify")
    return pin["md5"]


def _fingerprint(
    data_paths: list[str],
    settings: dict,
    extra_files: tuple[str, ...],
) -> dict:
    """Everything that defines this computation, as one flat dict. Settings enter verbatim,
    which keeps the .meta.json sidecars readable, and the data_hash_i/extra_hash_i split is
    sidecar labelling alone as both are file hashes."""
    meta: dict = {f"data_hash_{i}": file_md5(p) for i, p in enumerate(data_paths)}
    meta.update({f"extra_hash_{i}": source_md5(p) for i, p in enumerate(extra_files)})
    meta.update(settings)
    return meta


def _fingerprint_hash(
    data_paths: list[str],
    settings: dict,
    extra_files: tuple[str, ...],
) -> str:
    """Collapse the fingerprint to the 12-char id used as a filename, where sort_keys=True
    matters, else the same settings built in a different order would hash differently and miss a
    valid cache entry unreported."""
    fp = _fingerprint(data_paths, settings, extra_files)
    return hashlib.md5(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:12]


# Pattern 1: content-addressed.

def get_cache_path(
    name: str,
    data_paths: list[str],
    settings: dict,
    *,
    extra_files: tuple[str, ...] = (),
    cache_dir: str = CACHE_DIR,
    ext: str = "csv",
) -> str:
    """The content-addressed path for this computation, where any change to the inputs gives a new hash and old results are preserved."""
    h = _fingerprint_hash(data_paths, settings, extra_files)
    return os.path.join(cache_dir, name, f"{h}.{ext}")


def data_stem(path: str) -> str:
    """Filename stem of a data file, the data-identifying component of Pattern 2's fixed-path cache filenames."""
    return os.path.splitext(os.path.basename(path))[0]


# Pattern 2: fixed-path fingerprint validation.

def fingerprint_valid(
    path: str,
    data_paths: list[str],
    settings: dict,
    *,
    extra_files: tuple[str, ...] = (),
) -> bool:
    """True where path and its .meta.json sidecar exist and match this fingerprint, reading both the old flat and current nested sidecar formats."""
    sidecar = os.path.splitext(path)[0] + ".meta.json"
    if not os.path.exists(path) or not os.path.exists(sidecar):
        return False
    try:
        current = _fingerprint(data_paths, settings, extra_files)
    except OSError:
        return False
    with open(sidecar) as f:
        stored = json.load(f)
    return (stored["fingerprint"] if "fingerprint" in stored else stored) == current


def write_meta(
    cache_path: str,
    data_paths: list[str],
    settings: dict,
    *,
    name: str | None = None,
    extra_files: tuple[str, ...] = (),
) -> None:
    """Write a .meta.json sidecar beside cache_path, read by fingerprint_valid() and informational for content-addressed caches."""
    if name is None:
        name = os.path.basename(os.path.dirname(cache_path))
    meta = {
        "name": name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "data_files": data_paths,
        "settings": settings,
        "fingerprint": _fingerprint(data_paths, settings, extra_files),
    }
    sidecar = os.path.splitext(cache_path)[0] + ".meta.json"
    with open(sidecar, "w") as f:
        json.dump(meta, f, indent=2)


# Pattern 3: best-architecture pointer.

BEST_ARCHITECTURE_PATH = os.path.join(CACHE_DIR, "optuna", "best_architecture.json")


def save_best_architecture(
    layer_widths: list[int],
    activations: list[str],
    lr: float,
    *,
    val_mae: float,
    data_path: str,
    study_name: str,
    optimiser_name: str = "adam",
    n_params: int | None = None,
) -> None:
    """Persist the winning Optuna config as the best-architecture pointer, overwriting any
    previous one, whilst the full history stays in the study's SQLite file."""
    os.makedirs(os.path.dirname(BEST_ARCHITECTURE_PATH), exist_ok=True)
    with open(BEST_ARCHITECTURE_PATH, "w") as f:
        json.dump({
            "layer_widths": layer_widths,
            "activations": activations,
            "lr": lr,
            "optimiser_name": optimiser_name,
            "val_mae": val_mae,
            "n_params": n_params,
            "data_path": data_path,
            "study_name": study_name,
            "saved": datetime.datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)


def load_best_architecture() -> dict | None:
    """Load the current best-architecture pointer, or None where no search has saved one."""
    if not os.path.exists(BEST_ARCHITECTURE_PATH):
        return None
    with open(BEST_ARCHITECTURE_PATH) as f:
        return json.load(f)


def resolve_architecture(
    layer_widths: list[int],
    activations: list[str],
    lr: float,
    use_best_found: bool,
    optimiser_name: str = "adam",
) -> tuple[list[int], list[str], float, str]:
    """Return (layer_widths, activations, lr, optimiser_name) unchanged, called before
    computing a cache key since all four feed it.

    use_best_found=True raises (deprecated 2026-07-17), as the pointer is a side effect of
    whichever search ran last and was found stale, whilst every headline result passes its
    architecture explicitly in the plan YAMLs.
    Raising rather than falling back means a stale pointer cannot be applied unreported, so
    the fallback below is not reached.
    """
    if not use_best_found:
        return layer_widths, activations, lr, optimiser_name
    raise RuntimeError(
        "use_best_found=True is deprecated (2026-07-17): cache/optuna/best_architecture.json "
        "is a last-search side effect and was found stale ([32,32,16]/sgd from the 2026-07-12 "
        "multi-threshold study). Pass layer_widths/activations/lr/optimiser_name explicitly -- "
        "headline config (re-pinned 2026-07-23): layer_widths=[16,16,64], "
        "activations=[gelu,silu,leaky_relu], optimiser_name=adamw, lr=0.0014 "
        "(Optuna v5 trial 439, job 662170; previous pin [128,64,16]/leaky_relu,relu,silu/"
        "lr 0.0026 was trial 124, job 630579). See DECISIONS.md 2026-07-17 and 2026-07-23."
    )
    best = load_best_architecture()
    if best is None:
        print(
            "use_best_found=True but no cache/optuna/best_architecture.json found "
            "-- falling back to the given defaults. Run model.experiments.architecture_search() "
            "or optuna_search_multi_dataset() first."
        )
        return layer_widths, activations, lr, optimiser_name
    if "layer_widths" not in best:
        print(
            "use_best_found=True but cache/optuna/best_architecture.json is in the old "
            "BandGapNet schema (hidden_dim/n_layers) from before FlexNet became this "
            "project's only model class -- falling back to the given defaults. Run "
            "model.experiments.architecture_search() or optuna_search_multi_dataset() "
            "to produce a new FlexNet-schema pointer."
        )
        return layer_widths, activations, lr, optimiser_name
    best_optimiser = best.get("optimiser_name", "adam")
    print(
        f"Using best-found architecture: layer_widths={best['layer_widths']} "
        f"activations={best['activations']} lr={best['lr']:.2e} "
        f"optimiser={best_optimiser} (val MAE={best['val_mae']:.4f} eV, from study '{best['study_name']}')"
    )
    return best["layer_widths"], best["activations"], best["lr"], best_optimiser


def _row_count(path: str) -> int:
    """The number of data rows in a CSV, excluding the header."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def list_cached(name: str | None = None, cache_dir: str = CACHE_DIR) -> None:
    """Print all content-addressed cache entries with their settings, restricted to one cache type by name."""
    if not os.path.isdir(cache_dir):
        print(f"No cache directory found at '{cache_dir}'.")
        return

    subdirs = sorted(
        d for d in os.listdir(cache_dir)
        if os.path.isdir(os.path.join(cache_dir, d)) and (name is None or d == name)
    )
    if not subdirs:
        target = f"'{name}'" if name else f"'{cache_dir}'"
        print(f"No cached entries found under {target}.")
        return

    total = 0
    for subdir in subdirs:
        folder = os.path.join(cache_dir, subdir)
        entries = sorted(
            f for f in os.listdir(folder) if not f.endswith(".meta.json")
        )
        total += len(entries)
        noun = "entry" if len(entries) == 1 else "entries"
        print(f"\n{subdir}/  ({len(entries)} {noun})")
        for entry in entries:
            h = os.path.splitext(entry)[0]
            sidecar = os.path.join(folder, h + ".meta.json")
            entry_path = os.path.join(folder, entry)
            size_kb = os.path.getsize(entry_path) / 1024
            size_desc = f"{size_kb:.0f} KB"
            if entry.endswith(".csv"):
                size_desc += f", {_row_count(entry_path)} rows"
            if os.path.exists(sidecar):
                with open(sidecar) as f:
                    meta = json.load(f)
                created = meta.get("created", "unknown")
                data_files = meta.get("data_files", [])
                settings = meta.get("settings", {})
                print(f"  [{h}]  {created}  ({size_desc})")
                for p in data_files:
                    line = f"    data:  {p}"
                    # Surface an upstream entry's own settings inline, such as the MD-HIT threshold
                    sub_sidecar = os.path.splitext(p)[0] + ".meta.json"
                    if os.path.exists(sub_sidecar):
                        with open(sub_sidecar) as f:
                            sub_settings = json.load(f).get("settings", {})
                        if "threshold" in sub_settings:
                            similarity = sub_settings.get("similarity", "?")
                            line += f"  (MD-HIT threshold={sub_settings['threshold']}, similarity={similarity})"
                    print(line)
                for k, v in settings.items():
                    print(f"    {k}:  {v}")
            else:
                print(f"  [{h}]  (no metadata)  ({size_desc})")

    noun = "entry" if total == 1 else "entries"
    print(f"\n{total} cached {noun} across {len(subdirs)} type(s) in '{cache_dir}/'")


_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"


def print_table(df: pd.DataFrame, columns: list[str], numeric_cols: "set[str] | None" = None) -> None:
    """Print df[columns] as a fixed-width table, each cell green where it is unchanged against
    the row above and red where it changed. Floats print in scientific notation, so outlier
    magnitudes do not widen the column, and numeric_cols are right-justified."""
    if numeric_cols is None:
        numeric_cols = {"mae", "rmse", "r2", "n_params"}
    str_cols = {c: df[c].apply(lambda v: "NaN" if pd.isna(v) else (f"{v:.4e}" if isinstance(v, float) else str(v))) for c in columns}
    widths = {c: max(len(c), str_cols[c].str.len().max() if len(df) else 0) for c in columns}

    header = "  ".join(c.rjust(widths[c]) if c in numeric_cols else c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))

    prev_row: "dict[str, str] | None" = None
    for i in range(len(df)):
        cells = []
        for c in columns:
            text = str_cols[c].iloc[i]
            padded = text.rjust(widths[c]) if c in numeric_cols else text.ljust(widths[c])
            if prev_row is None:
                cells.append(padded)
            elif text == prev_row[c]:
                cells.append(f"{_ANSI_GREEN}{padded}{_ANSI_RESET}")
            else:
                cells.append(f"{_ANSI_RED}{padded}{_ANSI_RESET}")
        print("  ".join(cells))
        prev_row = {c: str_cols[c].iloc[i] for c in columns}


_DATASET_CACHE_KINDS = {
    "mdhit": "filtered_data",
    "hull":  "stability_filtered",
}


def list_datasets(kind: str, cache_dir: str = CACHE_DIR, sort_by: str = "created", ascending: bool = True, top: "int | None" = None) -> pd.DataFrame:
    """A sortable table of cached MD-HIT-filtered or hull-filtered datasets, one row per cache
    entry, for print_table(). Hull rows also surface the upstream mdhit_threshold and
    mdhit_similarity where chained on top of an MD-HIT-filtered input."""
    subdir = _DATASET_CACHE_KINDS[kind]
    folder = os.path.join(cache_dir, subdir)
    if not os.path.isdir(folder):
        return pd.DataFrame(columns=["hash", "created", "n_entries", "source"])

    rows = []
    for entry in sorted(f for f in os.listdir(folder) if f.endswith(".csv")):
        h = os.path.splitext(entry)[0]
        entry_path = os.path.join(folder, entry)
        sidecar = os.path.join(folder, h + ".meta.json")
        if not os.path.exists(sidecar):
            continue
        with open(sidecar) as f:
            meta = json.load(f)
        settings = meta.get("settings", {})
        data_files = meta.get("data_files", [])
        row = {
            "hash": h,
            "created": meta.get("created", "unknown"),
            "n_entries": _row_count(entry_path),
            "source": os.path.basename(data_files[0]) if data_files else "",
        }
        if kind == "mdhit":
            row["threshold"] = settings.get("threshold")
            row["similarity"] = settings.get("similarity")
        else:  # hull
            row["max_e_above_hull"] = settings.get("max_e_above_hull")
            upstream_sidecar = os.path.splitext(data_files[0])[0] + ".meta.json" if data_files else ""
            row["mdhit_threshold"] = ""
            row["mdhit_similarity"] = ""
            if upstream_sidecar and os.path.exists(upstream_sidecar):
                with open(upstream_sidecar) as f:
                    upstream_settings = json.load(f).get("settings", {})
                if "threshold" in upstream_settings:
                    row["mdhit_threshold"] = upstream_settings["threshold"]
                    row["mdhit_similarity"] = upstream_settings.get("similarity", "?")
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
        if top is not None:
            df = df.head(top)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect content-addressed cache entries")
    parser.add_argument(
        "name",
        nargs="?",
        help="Cache type to inspect (e.g. magpie_descriptors); omit for all. Ignored if --kind is given.",
    )
    parser.add_argument("--cache-dir", default=CACHE_DIR, metavar="DIR")
    parser.add_argument("--kind", choices=["mdhit", "hull"], default=None,
                         help="Print MD-HIT-filtered (cache/filtered_data/) or hull-energy-filtered "
                              "(cache/stability_filtered/) datasets as a sortable table, styled like "
                              "model.experiments --kind list_configs, instead of the raw per-entry dump.")
    parser.add_argument("--sort-by", default="created", help="Used only with --kind mdhit/hull: column to sort by (created, n_entries, threshold, similarity, max_e_above_hull)")
    parser.add_argument("--descending", action="store_false", dest="ascending", help="Used only with --kind mdhit/hull: sort highest-first instead of lowest-first")
    parser.add_argument("--top", type=int, default=None, help="Used only with --kind mdhit/hull: show only the first N rows")
    args = parser.parse_args()

    if args.kind is not None:
        df = list_datasets(args.kind, args.cache_dir, sort_by=args.sort_by, ascending=args.ascending, top=args.top)
        if args.kind == "mdhit":
            cols = ["hash", "created", "n_entries", "threshold", "similarity", "source"]
            numeric_cols = {"n_entries", "threshold"}
        else:
            cols = ["hash", "created", "n_entries", "max_e_above_hull", "mdhit_threshold", "mdhit_similarity", "source"]
            numeric_cols = {"n_entries", "max_e_above_hull", "mdhit_threshold"}
        if len(df):
            print_table(df, cols, numeric_cols)
        else:
            print(f"No cached '{_DATASET_CACHE_KINDS[args.kind]}' datasets found under {args.cache_dir}/.")
    else:
        list_cached(args.name, args.cache_dir)
