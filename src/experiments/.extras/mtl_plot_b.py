"""Panel B: distinguish the AE ceiling from loss in rapid MTL storage."""

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
TRAIN_NOISE_LEVELS = (0.00, 0.02, 0.04, 0.05)
TEST_NOISE_LEVELS = (0.00, 0.03, 0.06, 0.09)
REPETITIONS = 8
SEED = 73100
OUTPUT_STEM = "src/media/mtl_plot_b"


def main(show: bool = True):
    rows = []
    for plasticity in PLASTICITY_VARIANTS:
        for train_noise in TRAIN_NOISE_LEVELS:
            for test_noise in TEST_NOISE_LEVELS:
                try:
                    rows.extend(evaluate_condition(
                        dim_ca1=DIM_CA1,
                        num_cue_patterns=NUM_CUE_PATTERNS,
                        train_noise=train_noise,
                        test_noise=test_noise,
                        bit_kind=BIT_KIND,
                        plasticity=plasticity,
                        repetitions=REPETITIONS,
                        seed=SEED,
                    ))
                except FileNotFoundError as error:
                    print(f"[skip] {error}")

    fig, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    for plasticity in PLASTICITY_VARIANTS:
        selected = [row for row in rows if row["plasticity"] == plasticity]
        if not selected:
            continue
        grouped = {}
        for row in selected:
            key = (row["train_noise"], row["test_noise"])
            grouped.setdefault(key, []).append(row)
        x = [np.mean([item["ae_cosine"] for item in values])
             for values in grouped.values()]
        y = [np.mean([item["mtl_cosine"] for item in values])
             for values in grouped.values()]
        axis.scatter(
            x, y, s=48, alpha=0.8, color=RULE_COLORS[plasticity],
            label=RULE_LABELS[plasticity], edgecolor="white", linewidth=0.5,
        )

    limits = axis.get_xlim(), axis.get_ylim()
    low = min(limits[0][0], limits[1][0])
    high = max(limits[0][1], limits[1][1])
    axis.plot([low, high], [low, high], linestyle="--", color="0.45",
              linewidth=1.2, label="AE ceiling")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Direct AE output–target cosine")
    axis.set_ylabel("Post-MTL output–target cosine")
    axis.set_title("Separating representation quality from rapid-memory recall")
    axis.legend(frameon=False, fontsize=9)
    style_axis(axis)

    save_rows(rows, OUTPUT_STEM + ".csv")
    save_figure(fig, OUTPUT_STEM)
    if show:
        plt.show()
    return fig, axis, rows


if __name__ == "__main__":
    main()
