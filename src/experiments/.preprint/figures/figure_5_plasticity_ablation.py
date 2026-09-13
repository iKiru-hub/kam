"""Build the paired base-versus-error-driven plasticity comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.preprint.artifacts import load_arrays
from experiments.preprint.figures.common import mean_sem


LABELS = {"base": "Direct instructed-write", "err2": "Bounded error-driven"}


def build(artifact: Path, output: Path) -> Path:
    arrays = load_arrays(artifact)
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True)
    for rule_index, rule in enumerate(arrays["rules"].tolist()):
        for axis, metric, label in zip(axes, ("output_cosine", "identity"), ("Output-target cosine", "Nearest-target accuracy")):
            values = arrays[metric][:, rule_index].mean(axis=-1)
            mean, sem = mean_sem(values)
            axis.plot(arrays["fractions"], mean, marker="o", label=LABELS.get(rule, rule))
            axis.fill_between(arrays["fractions"], mean - sem, mean + sem, alpha=0.15)
            axis.set(xlabel="Dropped input fraction", ylabel=label, ylim=(-0.04, 1.04))
    axes[-1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Plasticity-rule comparison under matched parameters")
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
