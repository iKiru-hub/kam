"""Build corruption robustness and CA3-key ablation plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.preprint.artifacts import load_arrays
from experiments.preprint.figures.common import mean_sem


def build(artifact: Path, output: Path) -> Path:
    arrays = load_arrays(artifact)
    modes = arrays["key_modes"].tolist()
    fractions = arrays["fractions"]
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    for index, mode in enumerate(modes):
        for axis, metric, label in zip(
            axes,
            ("output_cosine", "identity", "key_cosine"),
            ("Output-target cosine", "Nearest-target accuracy", "Clean-corrupted CA3 cosine"),
        ):
            values = arrays[metric][:, index]
            mean, sem = mean_sem(values.reshape(values.shape[0], values.shape[1], -1).mean(axis=-1))
            axis.plot(fractions, mean, marker="o", label=mode)
            axis.fill_between(fractions, mean - sem, mean + sem, alpha=0.15)
            axis.set(xlabel="Dropped input fraction", ylabel=label, ylim=(-0.04, 1.04))
    axes[-1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Frozen recall under input corruption: CA3 key-map ablations")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build(args.artifact, args.output))
