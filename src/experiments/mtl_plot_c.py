"""Panel C: compare ET–IS and ET–(IS−X) rule families across corruption."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")

from experiments.mtl_plot_common import (
    PLASTICITY_VARIANTS,
    RULE_COLORS,
    RULE_LABELS,
    evaluate_condition,
    save_figure,
    save_rows,
    style_axis,
)


DIM_CA1 = 50
NUM_CUE_PATTERNS = 5
BIT_KIND = 0
TRAIN_NOISE_LEVEL = 0.05
TEST_NOISE_LEVELS = tuple(np.around(np.linspace(0.0, 0.10, 11), 2))
REPETITIONS = 12
SEED = 73200
OUTPUT_STEM = "src/media/mtl_plot_c"


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    if len(values) < 2:
        return mean, mean, mean
    half = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(len(values))
    return mean, mean - half, mean + half


def main(show: bool = True):
    rows = []
    for plasticity in PLASTICITY_VARIANTS:
        for test_noise in TEST_NOISE_LEVELS:
            try:
                rows.extend(evaluate_condition(
                    dim_ca1=DIM_CA1,
                    num_cue_patterns=NUM_CUE_PATTERNS,
                    train_noise=TRAIN_NOISE_LEVEL,
                    test_noise=float(test_noise),
                    bit_kind=BIT_KIND,
                    plasticity=plasticity,
                    repetitions=REPETITIONS,
                    seed=SEED,
                ))
            except FileNotFoundError as error:
                print(f"[skip] {error}")

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True,
                             constrained_layout=True)
    by_rule = {}
    for plasticity in PLASTICITY_VARIANTS:
        matrix = []
        for repetition in range(REPETITIONS):
            matrix.append([
                next(row["mtl_cosine"] for row in rows
                     if row["plasticity"] == plasticity
                     and row["repetition"] == repetition
                     and np.isclose(row["test_noise"], noise))
                for noise in TEST_NOISE_LEVELS
            ])
        by_rule[plasticity] = np.asarray(matrix)
        mean, low, high = _mean_ci(matrix)
        axes[0].plot(TEST_NOISE_LEVELS, mean, marker="o", linewidth=2,
                     color=RULE_COLORS[plasticity],
                     label=RULE_LABELS[plasticity])
        axes[0].fill_between(TEST_NOISE_LEVELS, low, high,
                             color=RULE_COLORS[plasticity], alpha=0.15)

    for left, right, label, color in (
        ("base", "err2", "ERR2 − BASE", RULE_COLORS["err2"]),
        ("btsp", "xbtsp", "xBTSP − BTSP", RULE_COLORS["xbtsp"]),
    ):
        difference = by_rule[right] - by_rule[left]
        mean, low, high = _mean_ci(difference)
        axes[1].plot(TEST_NOISE_LEVELS, mean, marker="o", linewidth=2,
                     color=color, label=label)
        axes[1].fill_between(TEST_NOISE_LEVELS, low, high,
                             color=color, alpha=0.15)

    axes[0].set_ylabel("Output–clean-target cosine")
    axes[0].set_title(
        f"Rule robustness after storage (AE/MTL train noise = {TRAIN_NOISE_LEVEL:.2f})"
    )
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    axes[1].axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Test corruption fraction")
    axes[1].set_ylabel("Paired rule difference")
    axes[1].set_title("Benefit of the error-dependent term")
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        style_axis(axis)

    save_rows(rows, OUTPUT_STEM + ".csv")
    save_figure(fig, OUTPUT_STEM)
    if show:
        plt.show()
    return fig, axes, rows


if __name__ == "__main__":
    main()

