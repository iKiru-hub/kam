"""Build an E0 diagnostic figure exclusively from a saved artifact."""

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


def build(artifact_dir: Path, output_path: Path) -> dict[str, object]:
    artifact_dir = artifact_dir.resolve()
    output_path = output_path.resolve()
    arrays_path = artifact_dir / "arrays.npz"
    report_path = artifact_dir / "report.json"
    if not arrays_path.exists() or not report_path.exists():
        raise FileNotFoundError("Artifact must contain arrays.npz and report.json")
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    report = json.loads(report_path.read_text(encoding="utf-8"))

    names = arrays["condition_names"].tolist()
    cosine = arrays["metric_raw_cosine"]
    targets = arrays["inputs"]
    outputs = arrays["outputs"]
    colors = ["#2678b2", "#d95f02", "#1b9e77", "#7570b3", "#666666"]

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    axes[0].imshow(targets, aspect="auto", vmin=0, vmax=1, cmap="gray_r")
    axes[0].set_title("Stored targets")
    axes[0].set_xlabel("EC coordinate")
    axes[0].set_ylabel("Memory")

    for index, (name, color) in enumerate(zip(names, colors)):
        x = np.full(cosine.shape[1], index, dtype=float)
        offsets = np.linspace(-0.12, 0.12, cosine.shape[1])
        axes[1].scatter(x + offsets, cosine[index], s=22, color=color, alpha=0.8)
        axes[1].plot(index, cosine[index].mean(), marker="_", markersize=18, color="black")
    axes[1].set_xticks(range(len(names)), [name.replace("_", "\n") for name in names])
    axes[1].set_ylabel("Output–target cosine")
    axes[1].set_title("Saved query metrics")
    axes[1].set_ylim(-0.05, 1.05)

    axes[2].imshow(outputs[0], aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axes[2].set_title("Aligned recalled output")
    axes[2].set_xlabel("EC coordinate")
    axes[2].set_ylabel("Memory")
    figure.suptitle(
        f"E0 reproducibility diagnostic · seed {report['root_seed']}", fontsize=11
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    figure_source = output_path.with_suffix(".csv")
    with figure_source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "memory_index", "raw_cosine"])
        for condition_index, condition in enumerate(names):
            for memory_index, value in enumerate(cosine[condition_index]):
                writer.writerow([condition, memory_index, float(value)])
    return {
        "figure": str(output_path),
        "source_data": str(figure_source),
        "scientific_digest": report["scientific_digest"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.artifact, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

