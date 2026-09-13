"""Draw the selective MEC/LEC degradation protocol for preprint Figure 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT_DIR / "article/figures/preprint/figure_1e.png"

MEC_COLOR = "#2678b2"
LEC_COLOR = "#e68613"
DROPPED_COLOR = "#e3e3e3"


def blend(color: str, strength: float) -> tuple[float, float, float]:
    base = np.asarray(to_rgb(color))
    return tuple(1.0 - float(strength) * (1.0 - base))


def draw_population(axis, x: float, y: float, values: np.ndarray, color: str,
                    dropped: set[int] | None = None) -> None:
    dropped = set() if dropped is None else dropped
    width, height, gap = 0.34, 0.32, 0.035
    for index, value in enumerate(values):
        left = x + index * (width + gap)
        if index in dropped:
            face = DROPPED_COLOR
        else:
            face = blend(color, 0.18 + 0.82 * float(value))
        patch = Rectangle((left, y), width, height, facecolor=face,
                          edgecolor="0.22", linewidth=0.75)
        axis.add_patch(patch)
        if index in dropped:
            pad = 0.075
            axis.plot([left + pad, left + width - pad],
                      [y + pad, y + height - pad], color="0.45", linewidth=1.2)
            axis.plot([left + pad, left + width - pad],
                      [y + height - pad, y + pad], color="0.45", linewidth=1.2)


def protocol_box(axis, x: float, text: str, color: str = "white") -> None:
    box = FancyBboxPatch((x, 3.25), 2.05, 0.47,
                         boxstyle="round,pad=0.04,rounding_size=0.05",
                         facecolor=color, edgecolor="0.2", linewidth=1.0)
    axis.add_patch(box)
    axis.text(x + 1.025, 3.485, text, ha="center", va="center", fontsize=10)


def build(output: Path) -> Path:
    figure, axis = plt.subplots(figsize=(9.2, 4.6))
    axis.set_xlim(0, 12.6)
    axis.set_ylim(0, 4.25)
    axis.axis("off")

    axis.text(6.3, 4.08, "Selective degradation during frozen recall",
              ha="center", va="center", fontsize=15, fontweight="bold")
    protocol_box(axis, 1.45, "store clean laps")
    protocol_box(axis, 5.25, "freeze plasticity", color="#f3f3f3")
    protocol_box(axis, 9.05, "probe degraded lap")
    for start in (3.60, 7.40):
        axis.annotate("", xy=(start + 1.25, 3.485), xytext=(start, 3.485),
                      arrowprops={"arrowstyle": "-|>", "color": "0.3", "lw": 1.4})

    mec_values = np.exp(-0.5 * ((np.arange(10) - 5.0) / 1.7) ** 2)
    lec_values = np.asarray([0.08, 0.08, 0.95, 1.0, 0.92, 0.10, 0.08, 0.08, 0.08, 0.08])
    mec_x, lec_x = 3.35, 7.45
    axis.text(mec_x + 1.86, 3.02, "MEC (spatial)", ha="center", fontsize=11, color=MEC_COLOR, fontweight="bold")
    axis.text(lec_x + 1.86, 3.02, "LEC (cue)", ha="center", fontsize=11, color=LEC_COLOR, fontweight="bold")

    rows = (
        (2.42, "Clean EC input", set(), set()),
        (1.55, "MEC degradation", {1, 3, 4, 7, 9}, set()),
        (0.68, "LEC degradation", set(), {0, 2, 3, 6, 8}),
    )
    for y, label, mec_drop, lec_drop in rows:
        axis.text(0.25, y + 0.16, label, ha="left", va="center", fontsize=11)
        draw_population(axis, mec_x, y, mec_values, MEC_COLOR, mec_drop)
        draw_population(axis, lec_x, y, lec_values, LEC_COLOR, lec_drop)
        axis.plot([7.10, 7.10], [y - 0.03, y + 0.35], color="0.4", linewidth=1.0)

    axis.text(6.3, 0.18,
              "Drop a fraction p of one EC population · mask fixed within a lap · new mask for each probe",
              ha="center", va="center", fontsize=9.5, color="0.25")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.output))


if __name__ == "__main__":
    main()
