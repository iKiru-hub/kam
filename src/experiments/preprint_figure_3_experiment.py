"""Assemble preprint Figure 3 from selective LEC and MEC degradation.

The main-text figure shows the modality-specific double dissociation using
normal sparse and dense CA3 key maps.  Shuffled and identity controls remain
available in the underlying artifacts and supplementary figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
ROOT_DIR = SRC_DIR.parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.preprint.artifacts import load_arrays
from experiments.preprint.figures.common import mean_sem


DEFAULT_LEC_RESULTS = ROOT_DIR / "results/mtl_cue_degradation/v1"
DEFAULT_MEC_RESULTS = ROOT_DIR / "results/mtl_mec_degradation/v1"
DEFAULT_OUTPUT = ROOT_DIR / "article/figures/preprint/figure_3_main.png"

DISPLAY_MODES = ("normal", "dense")
MODE_LABELS = {"normal": "Sparse CA3 key", "dense": "Dense CA3 key"}
MODE_COLORS = {"normal": "#2678b2", "dense": "#2ca02c"}


def add_panel_label(axis, label: str) -> None:
    axis.text(-0.13, 1.08, label, transform=axis.transAxes, fontsize=12, fontweight="bold", va="top")


def plot_metric(axis, arrays: dict[str, np.ndarray], metric: str, ylabel: str, chance: float | None = None) -> None:
    modes = arrays["key_modes"].tolist()
    fractions = arrays["fractions"]
    for mode in DISPLAY_MODES:
        mode_index = modes.index(mode)
        per_seed = arrays[metric][:, mode_index].mean(axis=-1)
        mean, sem = mean_sem(per_seed)
        axis.plot(fractions, mean, marker="o", linewidth=2, color=MODE_COLORS[mode], label=MODE_LABELS[mode])
        axis.fill_between(fractions, mean - sem, mean + sem, color=MODE_COLORS[mode], alpha=0.16)
    if chance is not None:
        axis.axhline(chance, color="0.55", linestyle=":", linewidth=1, label="Chance")
    axis.set(xlabel="Dropped input fraction", ylabel=ylabel, ylim=(-0.04, 1.04))
    axis.spines[["top", "right"]].set_visible(False)


def build(lec_artifact: Path, mec_artifact: Path, output: Path) -> Path:
    lec = load_arrays(lec_artifact)
    mec = load_arrays(mec_artifact)
    if not np.array_equal(lec["fractions"], mec["fractions"]):
        raise ValueError("LEC and MEC artifacts must use the same degradation fractions")

    figure, axes = plt.subplots(2, 2, figsize=(9.5, 7.4), sharex=True)
    plot_metric(axes[0, 0], lec, "cue_accuracy", "Cue identity accuracy", chance=0.5)
    axes[0, 0].set_title("LEC degradation: cue recall")
    add_panel_label(axes[0, 0], "A")

    plot_metric(axes[0, 1], lec, "mec_cosine", "MEC output-target cosine")
    axes[0, 1].set_title("LEC degradation: spatial output")
    add_panel_label(axes[0, 1], "B")

    chance_position = 1.0 / float(mec["example_target"].shape[-2])
    plot_metric(axes[1, 0], mec, "position_accuracy", "MEC nearest-position accuracy", chance=chance_position)
    axes[1, 0].set_title("MEC degradation: spatial recall")
    add_panel_label(axes[1, 0], "C")

    plot_metric(axes[1, 1], mec, "cue_accuracy", "Cue identity accuracy", chance=0.5)
    axes[1, 1].set_title("MEC degradation: cue recall")
    add_panel_label(axes[1, 1], "D")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Keep one shared legend; the dotted chance reference is self-explanatory.
    figure.suptitle("Selective sensory degradation dissociates cue and spatial recall", fontsize=14, y=0.98)
    figure.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.935))
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.83, wspace=0.25, hspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lec-results", type=Path, default=DEFAULT_LEC_RESULTS)
    parser.add_argument("--mec-results", type=Path, default=DEFAULT_MEC_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.lec_results, args.mec_results, args.output))


if __name__ == "__main__":
    main()
