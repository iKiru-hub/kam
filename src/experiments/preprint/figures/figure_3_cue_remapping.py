"""Build cue-context CA1 tuning summaries from saved arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.preprint.artifacts import load_arrays


def build(artifact: Path, output: Path) -> Path:
    arrays = load_arrays(artifact)
    # Units have no cross-seed identity.  A representative realization shows
    # actual fields; the population scatter retains all seeds for inference.
    seed_scores = np.column_stack((
        arrays["spatial_stability"].mean(axis=1),
        arrays["cue_modulation"].mean(axis=1),
    ))
    center = np.median(seed_scores, axis=0)
    seed_index = int(np.argmin(np.sum((seed_scores - center) ** 2, axis=1)))
    fields = arrays["probe_ca1"][seed_index]
    modulation = arrays["cue_modulation"][seed_index]
    order = np.argsort(np.argmax(fields.mean(axis=0), axis=0))
    vmax = float(np.quantile(fields, 0.97))

    figure = plt.figure(figsize=(12, 6.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1, 1, 1.15))
    heat_axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    trace_axis = figure.add_subplot(grid[1, :2])
    scatter_axis = figure.add_subplot(grid[:, 2])
    image = heat_axes[0].imshow(fields[0, :, order].T, aspect="auto", origin="lower", vmin=0, vmax=vmax, cmap="magma")
    heat_axes[0].set(title=f"Context A (seed {int(arrays['root_seeds'][seed_index])})", xlabel="Track position", ylabel="CA1 unit")
    heat_axes[1].imshow(fields[1, :, order].T, aspect="auto", origin="lower", vmin=0, vmax=vmax, cmap="magma")
    heat_axes[1].set(title="Context B, same unit order", xlabel="Track position", ylabel="CA1 unit")
    figure.colorbar(image, ax=heat_axes, label="CA1 activity", fraction=0.04)

    examples = np.argsort(modulation)[-3:][::-1]
    for unit in examples:
        trace_axis.plot(fields[0, :, unit], linewidth=1.8, label=f"unit {unit}: A")
        trace_axis.plot(fields[1, :, unit], linewidth=1.4, linestyle="--", label=f"unit {unit}: B")
    trace_axis.set(title="Cue-sensitive example fields", xlabel="Track position", ylabel="CA1 activity")
    trace_axis.legend(frameon=False, ncol=3, fontsize=7)

    scatter_axis.scatter(arrays["spatial_stability"].reshape(-1), arrays["cue_modulation"].reshape(-1), alpha=0.35, s=12)
    scatter_axis.axvline(0, color="0.75", linewidth=0.8)
    scatter_axis.set(xlabel="Spatial tuning stability", ylabel="Cue modulation", title="All units, all seeds")
    for axis in [*heat_axes, trace_axis, scatter_axis]:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Cue-sequence changes produce heterogeneous CA1 tuning")
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
