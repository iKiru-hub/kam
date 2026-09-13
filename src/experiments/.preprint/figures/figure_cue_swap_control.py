"""Plot the matched cue-swap and no-swap temporal control."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.preprint.artifacts import load_arrays
from experiments.preprint.figures.common import mean_sem


COLORS = {"swap": "#7b3294", "no_swap": "#4d4d4d"}
LABELS = {"swap": "Cue swaps", "no_swap": "Fixed cue order"}


def plot(axis, arrays: dict[str, np.ndarray]) -> None:
    """Draw the temporal control into an existing Matplotlib axis."""

    names = arrays["condition_names"][0].tolist()
    transitions = np.arange(2, arrays["transition_similarity"].shape[-1] + 2)
    for name in names:
        condition_index = names.index(name)
        mean, sem = mean_sem(arrays["transition_similarity"][:, condition_index])
        axis.plot(
            transitions,
            mean,
            color=COLORS[name],
            linewidth=2.0,
            marker="o",
            markersize=3.0,
            label=LABELS[name],
        )
        axis.fill_between(
            transitions,
            mean - sem,
            mean + sem,
            color=COLORS[name],
            alpha=0.16,
            linewidth=0,
        )

    swap_index = names.index("swap")
    event_laps = transitions[arrays["cue_changed"][0, swap_index]]
    for lap in event_laps:
        axis.axvline(lap, color="0.75", linestyle="--", linewidth=0.9)
    axis.set(
        xlabel="New lap",
        ylabel="Similarity to preceding lap",
        ylim=(-0.04, 1.04),
        title="Cue swaps trigger CA1 reconfiguration",
    )
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)


def build(artifact: Path, output: Path) -> Path:
    arrays = load_arrays(artifact)
    figure, axis = plt.subplots(figsize=(6.3, 3.8), constrained_layout=True)
    plot(axis, arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build(args.artifact, args.output))
