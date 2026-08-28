"""Build the central decoder-compatibility figure from a saved artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.preprint.artifacts import load_arrays
from experiments.preprint.figures.common import mean_sem


def build(artifact: Path, output: Path) -> Path:
    arrays = load_arrays(artifact)
    names = arrays["condition_names"].tolist()
    cosine = arrays["cosine"].mean(axis=-1)
    identity = arrays["identity"].mean(axis=-1)
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for axis, values, label in zip(axes[:2], (cosine, identity), ("Output-target cosine", "Memory identity accuracy")):
        mean, sem = mean_sem(values)
        axis.scatter(np.tile(np.arange(len(names)), len(values)), values.reshape(-1), color="#777777", alpha=0.45, s=16)
        axis.errorbar(range(len(names)), mean, yerr=sem, fmt="o", color="black", capsize=3)
        axis.set_xticks(range(len(names)), [name.replace("_", "\n") for name in names], fontsize=8)
        axis.set_ylabel(label)
        axis.set_ylim(-0.04, 1.04)
    fixed = names.index("fixed_permutation")
    aligned = names.index("aligned")
    for values in cosine:
        axes[2].plot([0, 1], values[[fixed, aligned]], color="#aaaaaa", linewidth=0.8)
    axes[2].scatter(np.zeros(len(cosine)), cosine[:, fixed], color="#d95f02")
    axes[2].scatter(np.ones(len(cosine)), cosine[:, aligned], color="#2678b2")
    axes[2].set_xticks([0, 1], ["Fixed permutation", "Aligned"])
    axes[2].set_ylabel("Output-target cosine")
    axes[2].set_ylim(-0.04, 1.04)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("One-shot recall requires decoder-compatible CA1 coordinates")
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
