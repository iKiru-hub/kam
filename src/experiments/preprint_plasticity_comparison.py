"""Summarize paired base-versus-err2 preprint simulations.

Run the two Figure-data scripts first.  This script reads their ``arrays.npz``
files and writes a compact four-panel comparison plus tidy seed-level and
summary CSV files.  It does not rerun any scientific simulation.

The comparison deliberately focuses on four pre-registered-style contrasts:

1. Figure 2A: aligned minus fixed-permutation reconstruction cosine;
2. Figure 2B: tuning similarity at swap transitions minus matched no-swap
   transitions;
3. Figure 3A: cue accuracy after 90% LEC degradation; and
4. Figure 3C: position accuracy after 90% MEC degradation.

Each line joins the two rules for one shared root seed.  Thus the plot shows
rule effects within identical encoders, EC inputs, CA3 wiring, and masks.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RULE_COLORS = {"base": "#4c78a8", "err2": "#e45756"}
RULE_LABELS = {"base": "Instructive-driven", "err2": "Error-driven"}


def read_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load one result artifact with pickle disabled."""

    with np.load(path / "arrays.npz", allow_pickle=False) as artifact:
        return {name: artifact[name] for name in artifact.files}


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def figure_2_metrics(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reduce Figure 2 arrays to rule x seed comparison metrics."""

    conditions = arrays["compatibility_conditions"].tolist()
    aligned = conditions.index("aligned")
    permuted = conditions.index("fixed_permutation")
    compatibility_cost = arrays["compatibility_cosine"][:, :, aligned] - arrays["compatibility_cosine"][:, :, permuted]

    # Schedule index 0 is the alternating cue-swap protocol.  For every seed,
    # compare its marked swap transitions with the same transition indices in
    # the matched no-swap schedule (schedule index 1).
    transitions = arrays["transition_similarity"]
    changed = arrays["cue_changed"][:, :, 0].astype(bool)
    swap_delta = np.empty(transitions.shape[:2], dtype=float)
    for rule_index in range(transitions.shape[0]):
        for seed_index in range(transitions.shape[1]):
            event = transitions[rule_index, seed_index, 0, changed[rule_index, seed_index]].mean()
            control = transitions[rule_index, seed_index, 1, changed[rule_index, seed_index]].mean()
            swap_delta[rule_index, seed_index] = event - control
    return {
        "compatibility_cost": compatibility_cost,
        "cue_swap_delta": swap_delta,
    }


def figure_3_metrics(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Extract primary sparse-key endpoints from Figure 3 arrays."""

    modes = arrays["key_modes"].tolist()
    normal = modes.index("normal")
    fractions = np.asarray(arrays["fractions"], dtype=float)
    matches = np.flatnonzero(np.isclose(fractions, 0.90))
    if len(matches) != 1:
        raise ValueError("Figure 3 arrays must contain exactly one 90% degradation level.")
    partial = int(matches[0])
    # Mean masks within seed before any comparison between rules.
    return {
        "lec_cue_accuracy_90": arrays["lec_cue_accuracy"][:, :, normal, partial].mean(axis=-1),
        "mec_position_accuracy_90": arrays["mec_position_accuracy"][:, :, normal, partial].mean(axis=-1),
    }


def plot_paired(axis, values: np.ndarray, rules: list[str], title: str, ylabel: str) -> None:
    """Show paired seed-level values, then the rule mean ± SEM."""

    positions = np.arange(len(rules))
    for seed_values in values.T:
        axis.plot(positions, seed_values, color="0.75", linewidth=0.8, alpha=0.8, zorder=1)
    for rule_index, rule in enumerate(rules):
        mean, sem = mean_sem(values[rule_index])
        color = RULE_COLORS.get(rule, "0.2")
        axis.scatter(np.full(values.shape[1], rule_index), values[rule_index], color=color, alpha=0.7, s=18, zorder=2)
        axis.errorbar(rule_index, mean, yerr=sem, color="black", marker="o", markersize=5, capsize=3, linewidth=1.2, zorder=3)
    axis.axhline(0.0, color="0.7", linestyle=":", linewidth=1)
    axis.set_xticks(positions, [RULE_LABELS.get(rule, rule) for rule in rules])
    axis.set(title=title, ylabel=ylabel)
    axis.spines[["top", "right"]].set_visible(False)


def write_tables(output: Path, rules: list[str], seeds: np.ndarray, metrics: dict[str, np.ndarray]) -> None:
    """Save both seed-level paired data and a compact rule-level summary."""

    output.mkdir(parents=True, exist_ok=True)
    seed_rows, summary_rows = [], []
    for metric, values in metrics.items():
        for rule_index, rule in enumerate(rules):
            mean, sem = mean_sem(values[rule_index])
            summary_rows.append({"metric": metric, "plasticity_rule": rule, "mean": mean, "sem": sem, "ci95_low": mean - 1.96 * sem, "ci95_high": mean + 1.96 * sem})
            seed_rows.extend({"metric": metric, "plasticity_rule": rule, "seed": int(seed), "value": float(value)} for seed, value in zip(seeds, values[rule_index]))
        if len(rules) == 2:
            difference = values[1] - values[0]
            mean, sem = mean_sem(difference)
            summary_rows.append({"metric": metric, "plasticity_rule": f"{rules[1]} - {rules[0]}", "mean": mean, "sem": sem, "ci95_low": mean - 1.96 * sem, "ci95_high": mean + 1.96 * sem})
    for name, rows in (("seed_level.csv", seed_rows), ("summary.csv", summary_rows)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-2", type=Path, default=ROOT_DIR / "results/preprint/figure_2")
    parser.add_argument("--figure-3", type=Path, default=ROOT_DIR / "results/preprint/figure_3")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "results/preprint/plasticity_comparison")
    args = parser.parse_args()

    figure_2, figure_3 = read_arrays(args.figure_2), read_arrays(args.figure_3)
    rules = figure_2["plasticity_rules"].tolist()
    if rules != figure_3["plasticity_rules"].tolist() or len(rules) != 2:
        raise ValueError("Figure 2 and Figure 3 must contain the same two plasticity rules.")
    if not np.array_equal(figure_2["root_seeds"], figure_3["root_seeds"]):
        raise ValueError("Figure 2 and Figure 3 must use the same paired root seeds.")

    metrics = {**figure_2_metrics(figure_2), **figure_3_metrics(figure_3)}
    labels = (
        ("compatibility_cost", "Readout compatibility", "Aligned − permuted cosine"),
        ("cue_swap_delta", "Cue-swap reconfiguration", "Swap − no-swap tuning similarity"),
        ("lec_cue_accuracy_90", "LEC degradation", "Cue accuracy at 90% loss"),
        ("mec_position_accuracy_90", "MEC degradation", "Position accuracy at 90% loss"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), constrained_layout=True)
    for axis, (metric, title, ylabel) in zip(axes.flat, labels):
        plot_paired(axis, metrics[metric], rules, title, ylabel)
    figure.suptitle("Paired comparison of instructive-driven and error-driven CA3→CA1 plasticity", fontsize=13)
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / "plasticity_comparison.png", dpi=300)
    figure.savefig(args.output / "plasticity_comparison.svg")
    plt.close(figure)
    write_tables(args.output, rules, figure_2["root_seeds"], metrics)
    print(args.output)


if __name__ == "__main__":
    main()
