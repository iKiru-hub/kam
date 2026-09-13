"""Plot the CA1 stability--cue-modulation distribution from saved arrays."""

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
    stability = arrays["spatial_stability"].reshape(-1)
    modulation = arrays["cue_modulation"].reshape(-1)

    figure, axis = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    axis.scatter(
        stability,
        modulation,
        alpha=0.32,
        s=18,
        color="#2678b2",
        edgecolors="none",
    )
    axis.axvline(0.0, color="0.75", linewidth=0.8)
    axis.set(
        xlabel="Spatial tuning stability",
        ylabel="Cue modulation",
        title="CA1 tuning across cue contexts",
    )
    axis.spines[["top", "right"]].set_visible(False)
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
