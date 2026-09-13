"""Generate the data behind preprint Figure 3.

Goal
    Test whether the stored association remains useful when either of the two
    EC input components is selectively degraded.  MEC-like units carry track
    position and LEC-like units carry the local cue identity.  This is a test
    of graceful partial-cue recall, not a claim of recurrent pattern
    completion: once a modality is made unidentifiable, the model cannot
    restore it.

Protocol in one replicate
    1. Pretrain the EC--CA1--EC encoder on clean MEC+LEC track samples.
    2. Train matched CA3--CA1 models over 20 clean laps with each selected
       update rule (``base`` and ``err2`` by default). The cue arrangement
       stays [0, 1] during storage.
    3. Freeze plasticity and probe the last clean lap repeatedly.  Each probe
       removes 0, 25, 50, 75, or 90 percent of *one* modality, using a mask
       fixed for that whole lap and resampled for the next probe.
    4. Compare the normal sparse CA3 key with a dense common-key control.  In
       the control every CA3 unit receives the same EC average, intentionally
       erasing item-specific key structure.

Panels A/B use LEC degradation:
    A: cue identity accuracy at the two cue positions.
    B: cosine similarity of the intact MEC output to its clean target.

Panels C/D use MEC degradation:
    C: nearest-position accuracy using only MEC output.
    D: cue identity accuracy from the intact LEC output.

The output directory contains ``arrays.npz``, ``config.json``, and
``source_data.csv``.  The four metric arrays correspond to panels A--D.
Their leading dimensions are ``(plasticity rule, root seed, key mode, ...)``.

    clean EC storage → freeze CA3→CA1 weights → selectively degrade an EC probe
                                                  ├─ MEC removed: test position
                                                  └─ LEC removed: test cue

    python src/experiments/preprint_figure_3.py
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import datagen
from experiments.preprint_common import (
    build_mtl,
    cue_track,
    row_cosine,
    run_mtl,
    train_autoencoder,
    write_artifact,
)


# Stable internal identifiers map to descriptive labels in manuscript figures.
RULE_LABELS = {"base": "Instructive-driven", "err2": "Error-driven"}


# All reported parameters are deliberately visible here.  The 20 seeds are
# independent model/track replicates; 12 masks measure within-seed variation.
# Both rules share each seed's EC laps, frozen encoder, and probe masks.
SETTINGS = {
    "seeds": list(range(51001, 51021)),
    "plasticity_rules": ["base", "err2"],
    "dimension": 50,
    "active": 5,
    "autoencoder": {
        "latent_dimension": 50,
        "beta_latent": 25.0,
        "beta_output": 25.0,
        "epochs": 256,
        "batch_size": 64,
        "learning_rate": 0.001,
    },
    "memory": {
        "ca3_dimension": 50,
        "ca3_inputs_per_unit": 29,
        "k_ca3": 3,
        "k_ca1": 5,
        "beta_ca3": 170.6,
        "beta_ca1": 41.8,
        "alpha": 0.092,
        "plasticity_rule": "err2",
    },
    "track": {
        "training_laps": 20,
        "validation_laps": 5,
        "lap_length": 50,
        "size": 50,
        "cue_positions": [10, 30],
        "cue_sigma": 4.0,
        "cue_beta": 40.0,
        "cue_alpha": 0.1,
        "mec_binarized": True,
        "mec_sigma": 5.0,
        "lec_sigma": 5.0,
    },
    "degradation": {
        "fractions": [0.0, 0.25, 0.5, 0.75, 0.9],
        "masks_per_fraction": 12,
        "key_modes": ["normal", "dense"],
    },
}


def seed_value(root_seed: int, stream: int) -> int:
    return int(np.random.SeedSequence([root_seed, stream]).generate_state(1)[0])


def apply_key_control(model, mode: str) -> None:
    """Leave the sparse key intact or make every CA3 key identical.

    The dense condition is not a sparsity-matched alternative architecture.
    It is a deliberately degenerate negative control: all CA3 units receive
    the same average EC input, so a distinct input cannot select a distinct
    retrieval key.
    """

    if mode == "normal":
        # The normal condition keeps the sparse, balanced EC→CA3 wiring
        # created in models.MTL.__init__.
        return
    if mode != "dense":
        raise ValueError(f"Unknown key mode: {mode}")
    dimension = model._dim_ei
    model.W_ei_ca3 = torch.nn.Parameter(torch.full((dimension, dimension), 1.0 / dimension))


def degrade(EC_values: np.ndarray, fraction: float, modality: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Zero a fixed subset of MEC or LEC coordinates for one whole probe lap.

    The mask is chosen across units, not independently at every track position.
    It therefore represents an input-population degradation rather than
    temporally flickering noise.  A fresh mask is used for every probe.
    """

    # EC input is ordered [MEC | LEC].  We sample unit indices from one half
    # and set those coordinates to zero at *every* track position in the lap.
    result = EC_values.copy()
    half = EC_values.shape[1] // 2
    candidates = np.arange(half) if modality == "mec" else np.arange(half, EC_values.shape[1])
    count = int(round(fraction * len(candidates)))
    mask = np.sort(rng.choice(candidates, size=count, replace=False)) if count else np.empty(0, dtype=int)
    result[:, mask] = 0.0
    return result, mask


def cue_accuracy(EC_recall: np.ndarray, cue_patterns: np.ndarray, positions: list[int]) -> float:
    """Classify each recalled cue trace by cosine similarity to LEC templates.

    A correct lap has cue 0 at the first cue position and cue 1 at the second.
    The score is therefore the fraction of these two position-level decisions
    that recover their original cue identity.
    """

    lec = EC_recall[np.asarray(positions), EC_recall.shape[1] // 2:]
    similarity = lec @ cue_patterns.T
    similarity /= np.maximum(np.linalg.norm(lec, axis=1)[:, None] * np.linalg.norm(cue_patterns, axis=1)[None, :], 1e-12)
    return float(np.mean(np.argmax(similarity, axis=1) == np.arange(len(positions))))


def position_accuracy(EC_recall: np.ndarray, EC_target: np.ndarray) -> float:
    """Decode each position from the recalled MEC pattern by nearest template.

    Every output MEC vector is compared with all 50 clean MEC vectors from the
    target lap.  Accuracy is the fraction whose nearest cosine match has the
    same track-position index; chance is therefore 1/50.
    """

    half = EC_recall.shape[1] // 2
    recalled, reference = EC_recall[:, :half], EC_target[:, :half]
    similarity = recalled @ reference.T
    similarity /= np.maximum(np.linalg.norm(recalled, axis=1)[:, None] * np.linalg.norm(reference, axis=1)[None, :], 1e-12)
    return float(np.mean(np.argmax(similarity, axis=1) == np.arange(len(EC_target))))


def prepare_seed(seed: int, settings: dict) -> dict:
    """Create clean EC storage, probes, and encoder shared by both rules.

    No CA3→CA1 weights are stored here.  The returned components are held
    fixed while each update rule gets its own matched CA3→CA1 model.
    """

    track = settings["track"]
    schedule = [[0, 1]] * track["training_laps"]
    EC_clean_laps = cue_track(track["training_laps"], track, schedule, seed_value(seed, 1))
    EC_validation = cue_track(track["validation_laps"], track, [[0, 1]] * track["validation_laps"], seed_value(seed, 2))
    autoencoder, encoder_mse = train_autoencoder(EC_clean_laps.reshape(-1, track["size"]), EC_validation.reshape(-1, track["size"]), settings, seed_value(seed, 3))
    return {
        "autoencoder": autoencoder,
        "encoder_mse": encoder_mse,
        "EC_clean_laps": EC_clean_laps,
        "EC_target": EC_clean_laps[-1],
        "cue_patterns": datagen.make_cues(2, track["size"] // 2, fixed=True),
    }


def run_rule(
    seed: int,
    prepared: dict,
    settings: dict,
    plasticity_rule: str,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Apply one update rule to the shared Figure-3 degradation protocol.

    A model is trained once per key mode and re-used for every degradation
    level and mask.  ``run_mtl(..., learn=False)`` freezes CA3--CA1 weights;
    degraded probes consequently test retrieval from the cleanly stored
    association rather than adaptation to damage.
    """

    track = settings["track"]
    degradation = settings["degradation"]
    memory = {**settings["memory"], "plasticity_rule": plasticity_rule}
    EC_clean_laps = prepared["EC_clean_laps"]
    EC_target = prepared["EC_target"]

    # ``values[metric][key mode, dropped fraction, mask]``.  Keeping masks
    # explicit lets plot code average within seed before calculating SEM over
    # seeds; masks are repeated probes, not independent model replicates.
    shape = (len(degradation["key_modes"]), len(degradation["fractions"]), degradation["masks_per_fraction"])
    values = {
        "lec_cue_accuracy": np.empty(shape, dtype=np.float32),
        "lec_mec_cosine": np.empty(shape, dtype=np.float32),
        "mec_position_accuracy": np.empty(shape, dtype=np.float32),
        "mec_cue_accuracy": np.empty(shape, dtype=np.float32),
    }
    rows = []
    for mode_index, mode in enumerate(degradation["key_modes"]):
        # ------------------------------------------------------------------
        # 2. Store clean laps once for this key architecture.
        #
        # ``normal`` has item-selective sparse keys. ``dense`` replaces the
        # EC→CA3 matrix with identical rows before storage, so all items share
        # essentially one common key.
        # ------------------------------------------------------------------
        # The wiring seed does not depend on rule.  Base and err2 therefore
        # start from the same CA3 keys before they diverge through learning.
        model = build_mtl(prepared["autoencoder"], memory, seed_value(seed, 10 + mode_index))
        apply_key_control(model, mode)
        # Storage happens on the same clean laps for all conditions.
        for EC_lap in EC_clean_laps:
            run_mtl(model, EC_lap, learn=True)
        for fraction_index, fraction in enumerate(degradation["fractions"]):
            for mask_index in range(degradation["masks_per_fraction"]):
                # ----------------------------------------------------------
                # 3. Probe the *same* frozen model under two complementary
                # forms of dropout.  The model is never retrained after a
                # mask is applied, which is why each metric reflects recall
                # from clean storage rather than compensation for damage.
                # ----------------------------------------------------------
                for modality in ("lec", "mec"):
                    rng = np.random.default_rng(np.random.SeedSequence([seed, mode_index, fraction_index, mask_index, 0 if modality == "lec" else 1]))
                    EC_probe, mask = degrade(EC_target, fraction, modality, rng)
                    EC_recall, _, _ = run_mtl(model, EC_probe, learn=False)
                    if modality == "lec":
                        # LEC loss: the direct prediction is cue identity at
                        # the two cue locations.  MEC cosine is the parallel
                        # check that the untouched spatial output persists.
                        measurements = {
                            "lec_cue_accuracy": cue_accuracy(EC_recall, prepared["cue_patterns"], track["cue_positions"]),
                            "lec_mec_cosine": float(row_cosine(EC_recall[:, :track["size"] // 2], EC_target[:, :track["size"] // 2]).mean()),
                        }
                    else:
                        # MEC loss: decode position from MEC output.  Cue
                        # identity from the untouched LEC output is the
                        # complementary control measurement.
                        measurements = {
                            "mec_position_accuracy": position_accuracy(EC_recall, EC_target),
                            "mec_cue_accuracy": cue_accuracy(EC_recall, prepared["cue_patterns"], track["cue_positions"]),
                        }
                    for metric, value in measurements.items():
                        values[metric][mode_index, fraction_index, mask_index] = value
                    rows.append({
                        "experiment": f"{modality}_degradation",
                        "seed": seed,
                        "encoder_validation_mse": prepared["encoder_mse"],
                        "plasticity_rule": plasticity_rule,
                        "key_mode": mode,
                        "dropped_fraction": fraction,
                        "mask_index": mask_index,
                        "dropped_units": len(mask),
                        **measurements,
                    })
    return values, rows


def mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the across-seed mean and SEM along the first axis."""

    values = np.asarray(values, dtype=float)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def save_panel(figure, output: Path, name: str, rule: str, display_rule: str) -> None:
    """Save a rule-specific panel and an unsuffixed primary-rule copy."""

    figure.savefig(output / f"{name}_{rule}.png", dpi=300, bbox_inches="tight")
    if rule == display_rule:
        figure.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_degradation_panel(
    values: np.ndarray,
    fractions: np.ndarray,
    modes: list[str],
    title: str,
    ylabel: str,
    chance: float | None = None,
):
    """Plot one metric after averaging repeated masks within each seed."""

    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    colors = {"normal": "#2678b2", "dense": "#2ca02c"}
    for mode_index, mode in enumerate(modes):
        seed_values = values[:, mode_index].mean(axis=-1)
        mean, sem = mean_sem(seed_values)
        axis.plot(fractions, mean, marker="o", color=colors.get(mode, "0.2"), label=f"{mode} CA3 key", linewidth=2)
        axis.fill_between(fractions, mean - sem, mean + sem, color=colors.get(mode, "0.2"), alpha=0.16)
    if chance is not None:
        axis.axhline(chance, color="0.45", linestyle=":", linewidth=1, label="chance")
    axis.set(xlabel="Dropped input fraction", ylabel=ylabel, ylim=(-0.04, 1.04), title=title)
    axis.legend(frameon=False, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    return figure


def build_panels(arrays: dict[str, np.ndarray], output: Path, display_rule: str) -> None:
    """Save Figure-3 panels as individual PNGs for external composition.

    ``plot_2_a.png`` through ``plot_2_d.png`` use ``display_rule``. Matching
    ``_base`` and ``_err2`` files retain both update-rule variants.
    """

    rules = arrays["plasticity_rules"].tolist()
    modes = arrays["key_modes"].tolist()
    fractions = arrays["fractions"]
    panel_specs = (
        ("plot_2_a", "lec_cue_accuracy", "LEC degradation: cue recall", "Cue identity accuracy", 0.5),
        ("plot_2_b", "lec_mec_cosine", "LEC degradation: spatial output", "MEC output–target cosine", None),
        ("plot_2_c", "mec_position_accuracy", "MEC degradation: spatial recall", "MEC nearest-position accuracy", 1.0 / 50.0),
        ("plot_2_d", "mec_cue_accuracy", "MEC degradation: cue recall", "Cue identity accuracy", 0.5),
    )
    for rule_index, rule in enumerate(rules):
        rule_label = RULE_LABELS.get(rule, rule)
        for name, metric, title, ylabel, chance in panel_specs:
            figure = plot_degradation_panel(arrays[metric][rule_index], fractions, modes, f"{title} ({rule_label})", ylabel, chance)
            save_panel(figure, output, name, rule, display_rule)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "results/preprint/figure_3")
    parser.add_argument("--rules", nargs="+", choices=("base", "err2"), default=SETTINGS["plasticity_rules"], help="Plasticity rules to compare; both are paired by default.")
    parser.add_argument("--display-rule", default=None, help="Rule used for unsuffixed plot_2_*.png panels; defaults to err2 when present.")
    parser.add_argument("--quick", action="store_true", help="Run two low-cost seeds for a smoke test.")
    args = parser.parse_args()

    settings = copy.deepcopy(SETTINGS)
    settings["plasticity_rules"] = list(args.rules)
    if args.quick:
        settings["seeds"] = settings["seeds"][:2]
        settings["autoencoder"]["epochs"] = 8
        settings["degradation"]["masks_per_fraction"] = 2

    # ------------------------------------------------------------------
    # 4. Repeat the full store-then-degrade protocol over independent seeds.
    #
    # Arrays retain masks within seed; plot code should average masks inside a
    # seed before using the 20 seed means for SEM or confidence intervals.
    # ------------------------------------------------------------------
    results, rows = [], []
    for index, seed in enumerate(settings["seeds"], start=1):
        print(f"Figure 3 seed {seed} ({index}/{len(settings['seeds'])})", flush=True)
        prepared = prepare_seed(seed, settings)
        rule_results = []
        for plasticity_rule in settings["plasticity_rules"]:
            result, seed_rows = run_rule(seed, prepared, settings, plasticity_rule)
            rule_results.append(result)
            rows.extend(seed_rows)
        results.append({name: np.stack([result[name] for result in rule_results]) for name in rule_results[0]})
    arrays = {
        "root_seeds": np.asarray(settings["seeds"]),
        "plasticity_rules": np.asarray(settings["plasticity_rules"]),
        "key_modes": np.asarray(settings["degradation"]["key_modes"]),
        "fractions": np.asarray(settings["degradation"]["fractions"], dtype=np.float32),
        **{name: np.stack([result[name] for result in results]).swapaxes(0, 1) for name in results[0]},
    }
    display_rule = args.display_rule or ("err2" if "err2" in settings["plasticity_rules"] else settings["plasticity_rules"][0])
    if display_rule not in settings["plasticity_rules"]:
        raise ValueError("--display-rule must be one of the selected --rules")
    artifact = write_artifact(args.output, settings, arrays, rows)
    build_panels(arrays, artifact, display_rule)
    print(artifact)


if __name__ == "__main__":
    main()
