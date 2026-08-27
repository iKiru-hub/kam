"""Render the development-only rule/rate selection from frozen arrays."""

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


def build(artifact_dir: Path, output_path: Path) -> dict[str, str]:
    artifact_dir = artifact_dir.resolve()
    output_path = output_path.resolve()
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    with np.load(artifact_dir / "arrays.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    rules = arrays["rule_names"].tolist()
    alphas = arrays["alphas"]
    conditions = arrays["condition_names"].tolist()
    cosine = arrays["metric_raw_cosine"].mean(axis=-1)
    shown = ["aligned", "fixed_permutation", "matched_decoder_rescue"]
    colors = {
        "aligned": "#2678b2",
        "fixed_permutation": "#d95f02",
        "matched_decoder_rescue": "#1b9e77",
    }
    figure, axes = plt.subplots(1, len(rules), figsize=(10.5, 4), sharey=True, constrained_layout=True)
    source_rows = []
    for rule_index, (rule, axis) in enumerate(zip(rules, np.atleast_1d(axes))):
        for condition in shown:
            condition_index = conditions.index(condition)
            values = cosine[rule_index, :, :, condition_index]
            means = values.mean(axis=1)
            sem = values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
            axis.errorbar(
                alphas,
                means,
                yerr=sem,
                marker="o",
                linewidth=1.7,
                capsize=3,
                color=colors[condition],
                label=condition.replace("_", " "),
            )
            for alpha_index, alpha in enumerate(alphas):
                for seed_index, seed in enumerate(arrays["root_seeds"]):
                    source_rows.append(
                        [rule, float(alpha), int(seed), condition, float(values[alpha_index, seed_index])]
                    )
        selected = report["selection"]["selected_rates"][rule]
        if selected is not None:
            axis.axvline(selected, color="black", linestyle="--", linewidth=1, alpha=0.6)
        axis.set_xscale("log", base=2)
        axis.set_xticks(alphas, [f"{value:g}" for value in alphas])
        axis.set_ylim(-0.05, 1.02)
        axis.set_xlabel("Learning rate α")
        axis.set_title(f"{rule.upper()} · selected α={selected:g}" if selected else rule.upper())
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean output–target cosine")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle("Development-only plasticity selection (mean ± SEM across 8 seeds)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    source_path = output_path.with_suffix(".csv")
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rule", "alpha", "root_seed", "condition", "mean_raw_cosine"])
        writer.writerows(source_rows)
    return {
        "figure": str(output_path),
        "source_data": str(source_path),
        "scientific_digest": report["scientific_digest"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.artifact, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
