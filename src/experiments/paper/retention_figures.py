"""Build E2 retention panels from frozen arrays only."""

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

from experiments.paper.final_figures import COLORS, LABELS, _mean_ci


def build(artifact_dir: Path, output_stem: Path) -> dict[str, str]:
    artifact_dir = artifact_dir.resolve()
    output_stem = output_stem.resolve()
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    with np.load(artifact_dir / "arrays.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    conditions = arrays["condition_names"].tolist()
    loads = arrays["evaluation_loads"]
    cosine = arrays["metric_raw_cosine"]
    identity = arrays["metric_identity_correct"]
    seeds = arrays["root_seeds"]
    indices = {name: conditions.index(name) for name in conditions}
    rng = np.random.default_rng(20260826)

    figure, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
    aligned_matrix = np.full(
        (len(loads), cosine.shape[-1]), np.nan, dtype=np.float64
    )
    for load_index, load in enumerate(loads):
        aligned_matrix[load_index, : int(load)] = cosine[
            :, indices["aligned"], load_index, : int(load)
        ].mean(axis=0)
    image = axes[0].imshow(
        aligned_matrix,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    axes[0].set_yticks(range(len(loads)), loads)
    axes[0].set_xlabel("Storage-order index")
    axes[0].set_ylabel("Total stored memories")
    axes[0].set_title("A  Aligned recall matrix")
    figure.colorbar(image, ax=axes[0], label="Output cosine", fraction=0.046)

    final_index = len(loads) - 1
    age = np.arange(int(loads[-1]) - 1, -1, -1)
    for condition in ("aligned", "fixed_permutation", "matched_decoder_rescue"):
        values = cosine[:, indices[condition], final_index, : int(loads[-1])]
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(seeds))
        axes[1].plot(age, mean, color=COLORS[condition], label=LABELS[condition].replace("\n", " "))
        axes[1].fill_between(age, mean - sem, mean + sem, color=COLORS[condition], alpha=0.15)
    axes[1].set_xlabel("Memory age (subsequent stores)")
    axes[1].set_ylabel("Output cosine")
    axes[1].set_ylim(-0.03, 1.02)
    axes[1].set_title("B  Retention after 40 stores")
    axes[1].legend(frameon=False, fontsize=7)

    endpoint_index = loads.tolist().index(28)
    endpoint = cosine[:, :, endpoint_index, :7].mean(axis=-1)
    for condition_index, condition in enumerate(conditions):
        values = endpoint[:, condition_index]
        axes[2].scatter(
            condition_index + rng.uniform(-0.1, 0.1, size=len(seeds)),
            values,
            color=COLORS[condition],
            alpha=0.7,
            s=20,
            edgecolor="none",
        )
        mean, low, high = _mean_ci(values)
        axes[2].errorbar(condition_index, mean, yerr=[[mean-low],[high-mean]], fmt="_", markersize=18, color="black", capsize=4)
    axes[2].set_xticks(range(len(conditions)), [LABELS[name] for name in conditions], fontsize=8)
    axes[2].set_ylim(-0.03, 1.02)
    axes[2].set_ylabel("Oldest-quartile cosine")
    axes[2].set_title("C  Frozen load-28 endpoint")

    threshold = 0.70
    capacity = np.sum(
        (cosine[:, :, final_index, : int(loads[-1])] >= threshold)
        & (identity[:, :, final_index, : int(loads[-1])] == 1.0),
        axis=-1,
    )
    for condition_index, condition in enumerate(conditions):
        values = capacity[:, condition_index]
        axes[3].scatter(
            condition_index + rng.uniform(-0.1, 0.1, size=len(seeds)), values,
            color=COLORS[condition], alpha=0.7, s=20, edgecolor="none",
        )
        mean, low, high = _mean_ci(values)
        axes[3].errorbar(condition_index, mean, yerr=[[mean-low],[high-mean]], fmt="_", markersize=18, color="black", capsize=4)
    axes[3].set_xticks(range(len(conditions)), [LABELS[name] for name in conditions], fontsize=8)
    axes[3].set_ylabel("Memories meeting capacity rule")
    axes[3].set_ylim(-1, 41)
    axes[3].set_title("D  Capacity after 40 stores")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Sequential associative retention depends on decoder-compatible instruction", fontsize=12)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    figure.savefig(png, dpi=300)
    figure.savefig(pdf)
    plt.close(figure)

    source = output_stem.with_suffix(".csv")
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["root_seed", "condition", "endpoint_oldest7_cosine", "capacity_load40"])
        for seed_index, seed in enumerate(seeds):
            for condition_index, condition in enumerate(conditions):
                writer.writerow(
                    [int(seed), condition, float(endpoint[seed_index, condition_index]), int(capacity[seed_index, condition_index])]
                )
    return {
        "png": str(png),
        "pdf": str(pdf),
        "source_data": str(source),
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
