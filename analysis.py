"""Loads and cleans the cached CSV, parses formulas to Compositions, computes descriptors and element stats, and plots their distributions."""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from pymatgen.core import Composition

from config import DIAGRAMS_FOLDER, REQUIRED_COLUMNS
from data.dataset_analysis import count_elements, element_summary
from data.preprocessing import default_filtered_data_path

from config import PINNED_DATASET as _DEFAULT_FILTERED_PATH
from features.descriptors import compute_element_stats, compute_magpie_descriptors
from visualisation import (
    plot_element_stats_histograms,
    plot_elements_by_atomic_number,
    plot_magpie_distributions,
    plot_periodic_table_heatmap,
    plot_total_atoms,
    plot_unique_elements,
    plot_valence_electrons_per_atom,
)

def load_data(data_path: str) -> pd.DataFrame:
    """Load the dataset CSV into a dataframe indexed by material_id."""
    return pd.read_csv(data_path, index_col="material_id")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing formula_pretty or band_gap (config.REQUIRED_COLUMNS)."""
    return df.dropna(subset=REQUIRED_COLUMNS)


def filter_nonmetals(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only nonmetals (band_gap > 0), mirroring Jung, Jung & Cole (2024)'s regression setup."""
    return df[df["band_gap"] > 0]


def parse_compositions(df: pd.DataFrame) -> list[Composition]:
    """Parse formula_pretty strings to Composition objects, dropping rows with no formula."""
    return [Composition(f) for f in df["formula_pretty"].dropna()]


def run_analysis(
    data_path: str,
    save_figures: bool = True,
    show_figures: bool = False,
    element_view: str = "heatmap",
) -> None:
    """Build the dataset overview figures for data_path, written to DIAGRAMS_FOLDER."""
    df = load_data(data_path)
    print(f"Loaded {len(df)} materials.")

    valid = clean_data(df)
    print(f"Removed {len(df) - len(valid)} materials.")

    compositions = parse_compositions(valid)  # validated rows only, so descriptors align with the labelled data

    # Counted once, both branches read the same counts.
    element_counts = count_elements(compositions)
    summary_df = element_summary(element_counts)

    # A 2x2 sized for \linewidth printing, as the old wide row scaled every font illegibly.
    # The coverage bar chart is always panel 4, and element_view only adds the heatmap figure.
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(2, 2, figure=fig)

    ax_elements = fig.add_subplot(gs[0, 0])
    ax_atoms = fig.add_subplot(gs[0, 1])
    ax_valence = fig.add_subplot(gs[1, 0])
    ax_periodic = fig.add_subplot(gs[1, 1])

    plot_unique_elements(compositions, ax_elements)
    plot_total_atoms(compositions, ax_atoms)
    plot_valence_electrons_per_atom(compositions, ax_valence)
    plot_elements_by_atomic_number(summary_df, ax_periodic)

    if element_view == "heatmap":
        plot_periodic_table_heatmap(element_counts, save_figures=save_figures)  # pymatviz creates its own figure

    plt.tight_layout()

    if save_figures:
        os.makedirs(DIAGRAMS_FOLDER, exist_ok=True)
        plt.savefig(f"{DIAGRAMS_FOLDER}/analysis_overview.svg", bbox_inches="tight")

    desc_df = compute_magpie_descriptors(compositions, data_path)
    plot_magpie_distributions(desc_df, save_figures=save_figures)

    element_stats_df = compute_element_stats(compositions, data_path)
    plot_element_stats_histograms(element_stats_df, save_figures=save_figures)

    if show_figures:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default=_DEFAULT_FILTERED_PATH)
    parser.add_argument("--no-save", action="store_false", dest="save_figures")
    parser.add_argument("--show", action="store_true", dest="show_figures")
    parser.add_argument("--element-view", choices=["heatmap", "bar"], default="heatmap")
    args = parser.parse_args()

    run_analysis(args.data_file, save_figures=args.save_figures, show_figures=args.show_figures, element_view=args.element_view)
