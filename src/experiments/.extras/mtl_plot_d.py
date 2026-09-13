"""Panel D: summarize saved evolution searches across representation regimes."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")

from experiments.mtl_plot_common import (
    PLASTICITY_VARIANTS,
    RULE_COLORS,
    RULE_LABELS,
    load_evolution_catalog,
    save_figure,
    save_rows,
    style_axis,
)


BIT_KIND = 0
OUTPUT_STEM = "src/media/mtl_plot_d"


def main(show: bool = True):
    rows = [row for row in load_evolution_catalog()
            if row["bit_kind"] == BIT_KIND]
    regimes = sorted({
        (int(row["dim_ca1"]), int(row["num_cue_patterns"]))
        for row in rows
        if row["dim_ca1"] is not None and row["num_cue_patterns"] is not None
    })
    fig, axes = plt.subplots(
        1, max(len(regimes), 1),
        figsize=(5.0 * max(len(regimes), 1), 4.5),
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )

    for axis, (dim_ca1, num_cues) in zip(axes[0], regimes):
        subset = [row for row in rows
                  if row["dim_ca1"] == dim_ca1
                  and row["num_cue_patterns"] == num_cues]
        for plasticity in PLASTICITY_VARIANTS:
            rule_rows = [row for row in subset
                         if row["plasticity"] == plasticity]
            if not rule_rows:
                continue
            grouped = {}
            for row in rule_rows:
                grouped.setdefault(float(row["train_noise"]), []).append(
                    float(row["fitness"])
                )
            noise = np.asarray(sorted(grouped))
            fitness = np.asarray([max(grouped[value]) for value in noise])
            axis.plot(noise, fitness, marker="o", linewidth=2,
                      color=RULE_COLORS[plasticity],
                      label=RULE_LABELS[plasticity])
        axis.set_xlabel("Evolution/training corruption")
        axis.set_title(f"CA1 dim. {dim_ca1}; {num_cues} cues")
        style_axis(axis)

    if regimes:
        axes[0, 0].set_ylabel("Best evolution fitness")
        axes[0, -1].legend(frameon=False, fontsize=9)
    fig.suptitle("Plasticity-rule optima depend on the representation regime")

    save_rows(rows, OUTPUT_STEM + ".csv")
    save_figure(fig, OUTPUT_STEM)
    if show:
        plt.show()
    return fig, axes, rows


if __name__ == "__main__":
    main()
