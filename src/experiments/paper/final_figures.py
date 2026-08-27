"""Build the final E1 decoder-compatibility figure from frozen arrays."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kam-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "aligned": "Aligned",
    "fixed_permutation": "Fixed\nperm.",
    "matched_decoder_rescue": "Rescue",
    "random_content_matched": "Random\nmatched",
    "no_plasticity": "No\nplasticity",
}

COLORS = {
    "aligned": "#2678b2",
    "fixed_permutation": "#d95f02",
    "matched_decoder_rescue": "#1b9e77",
    "random_content_matched": "#7570b3",
    "no_plasticity": "#666666",
}


def _mean_ci(values: np.ndarray, critical: float = 2.093024054408263) -> tuple[float, float, float]:
    mean = float(values.mean())
    half = float(critical * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def build(artifact_dir: Path, output_stem: Path) -> dict[str, str]:
    artifact_dir = artifact_dir.resolve()
    output_stem = output_stem.resolve()
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    with np.load(artifact_dir / "arrays.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    conditions = arrays["condition_names"].tolist()
    seeds = arrays["root_seeds"]
    cosine = arrays["metric_raw_cosine"].mean(axis=-1)
    identity = arrays["metric_identity_correct"].mean(axis=-1)
    indices = {name: conditions.index(name) for name in conditions}
    rng = np.random.default_rng(20260826)

    figure, axes = plt.subplots(1, 4, figsize=(14.5, 3.8), constrained_layout=True)
    for condition_index, condition in enumerate(conditions):
        values = cosine[:, condition_index]
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        axes[0].scatter(
            condition_index + jitter,
            values,
            s=22,
            alpha=0.70,
            color=COLORS[condition],
            edgecolor="none",
        )
        mean, low, high = _mean_ci(values)
        axes[0].errorbar(
            condition_index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="_",
            markersize=18,
            linewidth=2,
            capsize=4,
            color="black",
        )
    axes[0].set_xticks(range(len(conditions)), [LABELS[name] for name in conditions], fontsize=8)
    axes[0].set_ylabel("Mean output–target cosine")
    axes[0].set_ylim(-0.04, 1.02)
    axes[0].set_title("A  Output content")

    aligned = cosine[:, indices["aligned"]]
    fixed = cosine[:, indices["fixed_permutation"]]
    rescue = cosine[:, indices["matched_decoder_rescue"]]
    for seed_index in range(len(seeds)):
        axes[1].plot([0, 1], [fixed[seed_index], aligned[seed_index]], color="#9e9e9e", alpha=0.55, linewidth=0.8)
    axes[1].scatter(np.zeros(len(seeds)), fixed, color=COLORS["fixed_permutation"], s=22, zorder=3)
    axes[1].scatter(np.ones(len(seeds)), aligned, color=COLORS["aligned"], s=22, zorder=3)
    axes[1].set_xticks([0, 1], ["Fixed", "Aligned"])
    axes[1].set_ylim(-0.04, 1.02)
    axes[1].set_title("B  Paired causal effect")

    rescue_difference = rescue - aligned
    axes[2].axhspan(-0.02, 0.02, color="#1b9e77", alpha=0.13, label="equivalence margin")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].scatter(np.arange(1, len(seeds) + 1), rescue_difference, color=COLORS["matched_decoder_rescue"], s=22)
    mean, low, high = _mean_ci(rescue_difference, critical=1.729132811521367)
    axes[2].errorbar(len(seeds) + 1.5, mean, yerr=[[mean - low], [high - mean]], fmt="o", color="black", capsize=4)
    axes[2].set_xlim(0, len(seeds) + 2.5)
    axes[2].set_ylim(-0.025, 0.025)
    axes[2].set_xlabel("Final seed")
    axes[2].set_ylabel("Rescue − aligned cosine")
    axes[2].set_title("C  Readout-only rescue")

    for condition_index, condition in enumerate(conditions):
        values = identity[:, condition_index]
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        axes[3].scatter(condition_index + jitter, values, s=22, alpha=0.70, color=COLORS[condition], edgecolor="none")
        mean, low, high = _mean_ci(values)
        axes[3].errorbar(condition_index, mean, yerr=[[mean-low],[high-mean]], fmt="_", markersize=18, color="black", linewidth=2, capsize=4)
    axes[3].axhline(1 / arrays["inputs"].shape[1], color="black", linestyle=":", linewidth=1, label="chance")
    axes[3].set_xticks(range(len(conditions)), [LABELS[name] for name in conditions], fontsize=8)
    axes[3].set_ylim(-0.04, 1.02)
    axes[3].set_ylabel("Memory identity accuracy")
    axes[3].set_title("D  Identity retrieval")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Decoder-compatible instruction enables one-shot associative recall", fontsize=12)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)

    source_path = output_stem.with_suffix(".csv")
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["root_seed", "condition", "mean_raw_cosine", "identity_accuracy"])
        for seed_index, seed in enumerate(seeds):
            for condition_index, condition in enumerate(conditions):
                writer.writerow(
                    [
                        int(seed),
                        condition,
                        float(cosine[seed_index, condition_index]),
                        float(identity[seed_index, condition_index]),
                    ]
                )
    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "source_data": str(source_path),
        "scientific_digest": report["scientific_digest"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.artifact, args.output_stem), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
