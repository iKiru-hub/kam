"""Build controlled cue-count panels from frozen final arrays."""

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

from experiments.paper.final_figures import COLORS


SHOWN = ("aligned", "fixed_permutation", "matched_decoder_rescue", "no_plasticity")
SHORT = {
    "aligned": "Aligned",
    "fixed_permutation": "Fixed permutation",
    "matched_decoder_rescue": "Matched-decoder rescue",
    "no_plasticity": "No plasticity",
}


def _plot_metric(axis, values, counts, conditions, ylabel):
    for condition in SHOWN:
        index = conditions.index(condition)
        seed_means = []
        for count_index, count in enumerate(counts):
            item_count = 2 * int(count)
            seed_means.append(values[:, count_index, index, :item_count].mean(axis=-1))
        seed_means = np.stack(seed_means, axis=1)
        mean = seed_means.mean(axis=0)
        sem = seed_means.std(axis=0, ddof=1) / np.sqrt(len(seed_means))
        axis.plot(counts, mean, marker="o", color=COLORS[condition], label=SHORT[condition])
        axis.fill_between(counts, mean - sem, mean + sem, color=COLORS[condition], alpha=0.15)
    axis.set_xticks(counts)
    axis.set_xlabel("Number of cue identities")
    axis.set_ylabel(ylabel)
    axis.set_ylim(-0.03, 1.03)
    axis.spines[["top", "right"]].set_visible(False)


def build(artifact_dir: Path, output_stem: Path) -> dict[str, str]:
    artifact_dir = artifact_dir.resolve()
    output_stem = output_stem.resolve()
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    with np.load(artifact_dir / "arrays.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    counts = arrays["cue_counts"]
    conditions = arrays["condition_names"].tolist()
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    _plot_metric(
        axes[0], arrays["metric_raw_cosine"], counts, conditions,
        "Mean output–target cosine",
    )
    axes[0].set_title("A  Full content")
    _plot_metric(
        axes[1], arrays["cue_identity_correct"], counts, conditions,
        "Cue identity accuracy",
    )
    axes[1].plot(counts, 1 / counts, color="black", linestyle=":", linewidth=1.2, label="chance")
    axes[1].set_title("B  Sensory identity")
    _plot_metric(
        axes[2], arrays["position_correct"], counts, conditions,
        "Position accuracy",
    )
    axes[2].axhline(0.5, color="black", linestyle=":", linewidth=1.2, label="chance")
    axes[2].set_title("C  Position factor")
    axes[2].legend(frameon=False, fontsize=7, loc="lower left")
    figure.suptitle("Decoder-compatible recall scales to increasing cue-identity load", fontsize=12)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    figure.savefig(png, dpi=300)
    figure.savefig(pdf)
    plt.close(figure)
    source = output_stem.with_suffix(".csv")
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["root_seed", "cue_count", "condition", "mean_raw_cosine", "cue_identity_accuracy", "position_accuracy"]
        )
        for seed_index, seed in enumerate(arrays["root_seeds"]):
            for count_index, count in enumerate(counts):
                item_count = 2 * int(count)
                for condition_index, condition in enumerate(conditions):
                    writer.writerow(
                        [
                            int(seed), int(count), condition,
                            float(arrays["metric_raw_cosine"][seed_index, count_index, condition_index, :item_count].mean()),
                            float(arrays["cue_identity_correct"][seed_index, count_index, condition_index, :item_count].mean()),
                            float(arrays["position_correct"][seed_index, count_index, condition_index, :item_count].mean()),
                        ]
                    )
    return {
        "png": str(png), "pdf": str(pdf), "source_data": str(source),
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
