"""Assemble preprint Figure 2 from compatibility and cue-track artifacts.

The figure combines the primary decoder-coordinate result, a matched temporal
cue-swap control, and CA1 fields from a representative realization.
Simulations are not rerun; all panels are reconstructed from result artifacts.
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
from experiments.preprint.figures.figure_cue_swap_control import plot as plot_temporal_control


DEFAULT_COMPATIBILITY = ROOT_DIR / "results/preprint/v1/compatibility"
DEFAULT_REMAPPING = ROOT_DIR / "results/preprint/v1/cue_remapping"
DEFAULT_TEMPORAL_CONTROL = ROOT_DIR / "results/preprint/v1/cue_swap_control"
DEFAULT_OUTPUT = ROOT_DIR / "article/figures/preprint/figure_2_main.png"

CONDITION_LABELS = {
    "aligned": "Aligned",
    "fixed_permutation": "Fixed\npermutation",
    "matched_decoder": "Matched\ndecoder",
    "random_content": "Random\ncontent",
    "no_plasticity": "No\nplasticity",
}


def representative_fields(arrays: dict[str, np.ndarray]) -> tuple[int, np.ndarray, np.ndarray, float]:
    """Select the realization closest to median stability and modulation."""

    scores = np.column_stack((
        arrays["spatial_stability"].mean(axis=1),
        arrays["cue_modulation"].mean(axis=1),
    ))
    center = np.median(scores, axis=0)
    seed_index = int(np.argmin(np.sum((scores - center) ** 2, axis=1)))
    fields = arrays["probe_ca1"][seed_index]
    order = np.argsort(np.argmax(fields.mean(axis=0), axis=0))
    vmax = float(np.quantile(fields, 0.97))
    return seed_index, fields, order, vmax


def add_panel_label(axis, label: str) -> None:
    axis.text(-0.13, 1.08, label, transform=axis.transAxes, fontsize=12, fontweight="bold", va="top")


def build(
    compatibility_artifact: Path,
    remapping_artifact: Path,
    temporal_control_artifact: Path,
    output: Path,
) -> Path:
    compatibility = load_arrays(compatibility_artifact)
    remapping = load_arrays(remapping_artifact)
    temporal_control = load_arrays(temporal_control_artifact)
    names = compatibility["condition_names"].tolist()
    cosine = compatibility["cosine"].mean(axis=-1)
    means, sems = mean_sem(cosine)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    condition_axis, temporal_axis = axes[0]
    context_a_axis, context_b_axis = axes[1]

    positions = np.arange(len(names))
    for seed_index, values in enumerate(cosine):
        jitter = np.random.default_rng(7000 + seed_index).uniform(-0.08, 0.08, size=len(names))
        condition_axis.scatter(positions + jitter, values, color="0.55", alpha=0.4, s=13, linewidths=0)
    condition_axis.errorbar(positions, means, yerr=sems, fmt="o", color="black", markersize=6, capsize=3, linewidth=1.2)
    condition_axis.set_xticks(positions, [CONDITION_LABELS.get(name, name) for name in names], fontsize=8)
    condition_axis.set(ylabel="Output-target cosine", ylim=(-0.04, 1.04), title="Decoder-coordinate compatibility")
    add_panel_label(condition_axis, "A")

    plot_temporal_control(temporal_axis, temporal_control)
    add_panel_label(temporal_axis, "B")

    seed_index, fields, order, vmax = representative_fields(remapping)
    image = context_a_axis.imshow(fields[0, :, order].T, aspect="auto", origin="lower", vmin=0, vmax=vmax, cmap="magma")
    context_b_axis.imshow(fields[1, :, order].T, aspect="auto", origin="lower", vmin=0, vmax=vmax, cmap="magma")
    context_a_axis.set(title=f"Context A (seed {int(remapping['root_seeds'][seed_index])})", xlabel="Track position", ylabel="CA1 unit")
    context_b_axis.set(title="Context B, same unit order", xlabel="Track position", ylabel="CA1 unit")
    figure.colorbar(image, ax=[context_a_axis, context_b_axis], label="CA1 activity", fraction=0.035, pad=0.02)
    add_panel_label(context_a_axis, "C")
    add_panel_label(context_b_axis, "D")

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Decoder-compatible CA1 learning supports recall and cue-dependent reconfiguration", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compatibility", type=Path, default=DEFAULT_COMPATIBILITY)
    parser.add_argument("--remapping", type=Path, default=DEFAULT_REMAPPING)
    parser.add_argument(
        "--temporal-control",
        type=Path,
        default=DEFAULT_TEMPORAL_CONTROL,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(
        args.compatibility,
        args.remapping,
        args.temporal_control,
        args.output,
    ))


if __name__ == "__main__":
    main()
