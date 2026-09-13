"""Generate the data behind preprint Figure 2.

Read this file as two independent experiments that share the same core idea:
CA3 supplies a sparse retrieval key, CA1 supplies a learned association, and a
frozen CA1-to-EC decoder gives the recalled CA1 activity its meaning.

Panel A: decoder-coordinate compatibility
    A one-layer autoencoder is pretrained on sparse EC patterns and then
    frozen.  Twenty-eight new patterns are stored through CA3-to-CA1 weights
    under both the direct-write (``base``) and error-driven (``err2``) rules.
    The instructed CA1 target is either left in the decoder's coordinates,
    permuted while leaving the decoder fixed, or permuted together with the
    decoder.  This asks whether information is sufficient for recall, or
    whether the *coordinates* of the instructed CA1 state must be compatible
    with the fixed readout basis.

    EC input/target → fixed encoder → IS (instructed CA1 target)
             └────→ sparse CA3 key ──→ plastic CA3→CA1 weights → fixed decoder

Panels B--E: cue-dependent CA1 reconfiguration
    MEC-like spatial input and LEC-like cue input form a circular track.  Two
    cues remain at fixed positions, but their identities swap every ten laps.
    A matched no-swap control has the same random seed, MEC trajectory,
    pretrained encoder, EC→CA3 wiring, and learning parameters. Each rule gets
    its own matched CA3→CA1 model. After each learned lap the
    model is frozen and probed, so an abrupt change in tuning can be assigned
    to the cue schedule rather than to learning continuing during the probe.

    learn scheduled lap → freeze weights → probe context A and context B
                               ↑ repeat once per lap

The saved arrays map directly to figure panels:
    compatibility_cosine       panel A
    transition_similarity      panel B
    probe_ca1                  panels C and D
    spatial_stability,
    cue_modulation             panel E

Every numerical array begins with ``(plasticity rule, root seed, ...)`` so
base-versus-err2 comparisons are paired rather than independent experiments.

It writes ``arrays.npz``, ``config.json``, and a tidy ``source_data.csv``.
Those are the only files a separate figure-composition script needs.

Run from the repository root:

    python src/experiments/preprint_figure_2.py

Use ``--quick`` for a small smoke run before the full 20-seed experiment.
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

from core import functions
from experiments.preprint_common import (
    apply_evolved_parameters,
    build_mtl,
    cue_track,
    row_cosine,
    run_mtl,
    sparse_patterns,
    train_autoencoder,
    write_artifact,
)


# Stable internal identifiers map to descriptive labels in manuscript figures.
RULE_LABELS = {"base": "Instructive-driven", "err2": "Error-driven"}


# All fallback values are here rather than being hidden in a configuration
# hierarchy.  At runtime, available evolution artifacts replace these defaults
# with the selected autoencoder and rule-specific MTL parameters.  The two
# rules still share seeds, inputs, encoders, storage order, and probes.
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
    "compatibility": {
        "training_patterns": 2048,
        "validation_patterns": 256,
        "memories": 28,
        "epochs": 1024,
        "ca3_inputs_per_unit": 2,
        "k_ca3": 5,
        "k_ca1": 5,
        "beta_ca3": 200.0,
        "beta_ca1": 25.0,
        "beta_output": 25.0,
        "write_alpha": 0.08,
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
        "training_laps": 40,
        "validation_laps": 10,
        "lap_length": 50,
        "size": 50,
        "cue_positions": [10, 30],
        "swap_every": 10,
        "cue_sigma": 4.0,
        "cue_beta": 40.0,
        "cue_alpha": 0.1,
        "mec_binarized": True,
        "mec_sigma": 5.0,
        "lec_sigma": 5.0,
    },
}

COMPATIBILITY_CONDITIONS = (
    "aligned",
    "fixed_permutation",
    "matched_decoder",
    "random_content",
    "no_plasticity",
)


def seed_value(root_seed: int, stream: int) -> int:
    """ Give each part of a replicate an independent deterministic seed """
    return int(np.random.SeedSequence([root_seed, stream]).generate_state(1)[0])


def write_associations(CA3_keys: torch.Tensor, IS_targets: torch.Tensor,
    config: dict, plasticity_rule: str) -> torch.Tensor:
    """ Store CA3-key/IS associations with either preprint update rule.

    ``base`` is direct instructed writing. ``err2`` first recalls the current
    CA1 state, then applies bounded potentiation/depression to its residual
    error.  The equations match the corresponding branches in ``models.MTL``;
    keeping this short standalone version makes the fixed IS-permutation
    compatibility control explicit.
    """

    weights = torch.zeros((CA3_keys.shape[1], CA3_keys.shape[1]),
                          dtype=torch.float32)
    for CA3_key, IS in zip(CA3_keys, IS_targets):
        if plasticity_rule == "base":
            weights = (1.0 - config["write_alpha"] * IS[:, None]) * weights + \
                    config["write_alpha"] * IS[:, None] @ CA3_key[None, :]
        elif plasticity_rule == "err2":
            CA1_recall = functions.sparsemoid((weights @ CA3_key).reshape(1, -1),
                K=config["k_ca1"], beta=config["beta_ca1"],).reshape(-1)
            positive_error = torch.relu(IS - CA1_recall)
            negative_error = torch.relu(CA1_recall - IS)
            potentiation = config["write_alpha"] * \
                    (positive_error[:, None] @ CA3_key[None, :]) * (1.0 - weights)
            depression = config["write_alpha"] * \
                    (negative_error[:, None] @ CA3_key[None, :]) * weights
            weights = (weights + potentiation - depression).clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unknown plasticity rule: {plasticity_rule}")
    return weights


def sparse_ca3_keys(EC_inputs: torch.Tensor, rng: np.random.Generator,
                    config: dict) -> torch.Tensor:
    """ Create balanced sparse CA3 keys from EC inputs for panel A.

    Each CA3 unit receives two permuted EC inputs.  Repeating a permutation
    construction balances EC participation across the CA3 population, then a
    top-k nonlinearity turns that projection into the retrieval key.
    """

    dimension = EC_inputs.shape[1]
    weights = np.zeros((dimension, dimension), dtype=np.float32)
    for _ in range(config["ca3_inputs_per_unit"]):
        weights[np.arange(dimension), rng.permutation(dimension)] += 1.0 / dimension
    return functions.sparsemoid(EC_inputs @ torch.as_tensor(weights).T,
                                K=config["k_ca3"], beta=config["beta_ca3"])


def recall_direct(keys: torch.Tensor, weights: torch.Tensor,
                  decoder: torch.Tensor, config: dict) -> np.ndarray:
    """ Read the direct-write memory through the fixed CA1-to-EC decoder """

    ca1 = functions.sparsemoid(keys @ weights.T, K=config["k_ca1"], beta=config["beta_ca1"])
    return torch.sigmoid(config["beta_output"] * (ca1 @ decoder.T)).detach().numpy()


def prepare_compatibility(seed: int, settings: dict) -> dict:
    """ Create shared pretraining data and counterfactual IS targets.

    The returned data are deliberately rule-free.  Both ``base`` and ``err2``
    subsequently see exactly this encoder, EC memory set, IS permutation, and
    storage order.  CA3 keys are generated inside ``run_compatibility`` from a
    matched seed stream using each rule's evolved CA3 parameters.
    """

    config = settings["compatibility"]

    # ------------------------------------------------------------------
    # 1. Make three disjoint EC datasets.
    #
    # Training/validation patterns define the encoder's pretrained output
    # basis.  ``EC_memories`` are held out until after pretraining: they are the
    # novel patterns that the CA3→CA1 synapses must store in one shot.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed_value(seed, 1))
    EC_training, seen = sparse_patterns(config["training_patterns"],
                                        settings["dimension"], settings["active"], rng)
    EC_validation, seen = sparse_patterns(config["validation_patterns"],
                                          settings["dimension"], settings["active"], rng, seen)
    EC_memories, _ = sparse_patterns(config["memories"], settings["dimension"],
                                     settings["active"], rng, seen)
    ae_settings = {**settings, "autoencoder": {**settings["autoencoder"],
                                               "epochs": config["epochs"]}}
    autoencoder, _ = train_autoencoder(EC_training, EC_validation, ae_settings, seed_value(seed, 2))

    # ------------------------------------------------------------------
    # 2. Pass the held-out memories through the frozen encoder.
    #
    # ``IS`` is the CA1 state that the plastic association is
    # asked to recreate.  ``decoder`` is frozen across all conditions except
    # the explicit matched-decoder restoration control below.
    # ------------------------------------------------------------------
    EC = torch.as_tensor(EC_memories)
    encoder, decoder, _, _ = autoencoder.get_weights(bias=False)
    IS = functions.sparsemoid(EC @ encoder.detach().T, K=config["k_ca1"],
                              beta=settings["autoencoder"]["beta_latent"])
    permutation = np.roll(rng.permutation(settings["dimension"]), 1)
    content_permutation = np.roll(rng.permutation(len(EC_memories)), 1)
    order = rng.permutation(len(EC_memories))

    # ------------------------------------------------------------------
    # 3. Define paired counterfactuals.
    #
    # A fixed CA1 permutation preserves the vector values but moves every
    # value into a different coordinate.  If the decoder stays unchanged,
    # those coordinates no longer have their originally learned meaning.
    # ------------------------------------------------------------------
    IS_by_condition = {
        "aligned": IS,
        "fixed_permutation": IS[:, permutation],
        "matched_decoder": IS[:, permutation],
        "random_content": IS[content_permutation],
        "no_plasticity": IS,
    }
    return {
        "EC_memories": EC_memories,
        "decoder": decoder.detach(),
        "IS_by_condition": IS_by_condition,
        "order": order,
        "permutation": permutation,
    }


def run_compatibility(seed: int, prepared: dict, settings: dict,
    plasticity_rule: str, ) -> tuple[np.ndarray, list[dict]]:

    """ Apply one rule to the shared Figure-2A coordinate protocol """

    config = settings.get("compatibility_by_rule", {}).get(
        plasticity_rule, settings["compatibility"])
    EC_memories = prepared["EC_memories"]
    rng = np.random.default_rng(seed_value(seed, 3))
    CA3_keys = sparse_ca3_keys(torch.as_tensor(EC_memories), rng, config)
    cosine = np.zeros((len(COMPATIBILITY_CONDITIONS), len(EC_memories)),
                      dtype=np.float32)
    rows = []
    for condition_index, condition in enumerate(COMPATIBILITY_CONDITIONS):
        if condition == "no_plasticity":
            weights = torch.zeros((settings["dimension"], settings["dimension"]),
                                  dtype=torch.float32)
        else:
            weights = write_associations(CA3_keys[prepared["order"]],
                            prepared["IS_by_condition"][condition][prepared["order"]],
                            config, plasticity_rule,)
        # The matched-decoder control changes decoder columns by the same
        # fixed permutation used for IS.  This is the algebraic rescue test.
        readout_decoder = prepared["decoder"]
        if condition == "matched_decoder":
            readout_decoder = readout_decoder[:, prepared["permutation"]]
        EC_recall = recall_direct(CA3_keys, weights,
                                  readout_decoder, config)
        cosine[condition_index] = row_cosine(EC_recall, EC_memories)
        rows.extend(
            {
                "experiment": "compatibility",
                "seed": seed,
                "plasticity_rule": plasticity_rule,
                "condition": condition,
                "memory": memory,
                "output_target_cosine": float(value),
            }
            for memory, value in enumerate(cosine[condition_index])
        )
    return cosine, rows


def tuning_similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """ Return one mean-centered spatial-tuning similarity per CA1 unit.

    Inputs have shape ``(track position, CA1 unit)``.  Mean centering makes
    this a comparison of field shape/location rather than overall firing rate.
    It is therefore used for field stability, while cue modulation below is a
    raw mean absolute activity difference.
    """

    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    return row_cosine(first.T, second.T)


def context_schedule(laps: int, swap_every: int, swap: bool) -> list[list[int]]:
    """ Return cue identities at the two fixed cue positions for each lap """
    if not swap:
        return [[0, 1]] * laps
    return [[0, 1] if (lap // swap_every) % 2 == 0 else [1, 0] for lap in range(laps)]


def prepare_cue_remapping(seed: int, settings: dict) -> dict:
    """ Create shared cue-track inputs and one frozen encoder for both rules.

    ``EC_swap_laps`` and ``EC_no_swap_laps`` are generated from the same seed, so
    their MEC halves are checked to be identical.  Each schedule gets a fresh
    CA3--CA1 model, but both receive the same pretrained encoder.  After each
    training lap, learning is paused while both cue contexts are probed.

    No CA3→CA1 model is made here.  That is important: the returned EC laps,
    probes, and autoencoder are exactly shared by the ``base`` and ``err2``
    versions of this replicate.
    """

    track = settings["track"]

    # ------------------------------------------------------------------
    # 1. Build two schedules for the *same* virtual environment.
    #
    # [0, 1] means cue identity 0 occupies position 10 and identity 1
    # occupies position 30.  The swap schedule periodically reverses this;
    # the control never does.  A shared track seed makes their MEC halves
    # identical, so the schedule is the only designed difference.
    # ------------------------------------------------------------------
    swap_schedule = context_schedule(track["training_laps"], track["swap_every"], swap=True)
    no_swap_schedule = context_schedule(track["training_laps"], track["swap_every"], swap=False)
    track_seed = seed_value(seed, 10)
    EC_swap_laps = cue_track(track["training_laps"], track, swap_schedule, track_seed)
    EC_no_swap_laps = cue_track(track["training_laps"], track, no_swap_schedule, track_seed)
    mec_size = track["size"] // 2
    if not np.array_equal(EC_swap_laps[:, :, :mec_size], EC_no_swap_laps[:, :, :mec_size]):
        raise RuntimeError("The matched control changed the MEC trajectory.")

    # ------------------------------------------------------------------
    # 2. Pretrain one frozen EC→CA1→EC basis for both schedules.
    #
    # This deliberately puts the focus on what CA3→CA1 plasticity learns,
    # rather than allowing each schedule to acquire its own output basis.
    # ------------------------------------------------------------------
    EC_validation = cue_track(track["validation_laps"], track, [[0, 1]] * track["validation_laps"], seed_value(seed, 11))
    autoencoder, _ = train_autoencoder(EC_swap_laps.reshape(-1, track["size"]), EC_validation.reshape(-1, track["size"]), settings, seed_value(seed, 12))
    EC_probe_a = cue_track(1, track, [[0, 1]], seed_value(seed, 13))[0]
    EC_probe_b = cue_track(1, track, [[1, 0]], seed_value(seed, 13))[0]
    EC_probes = np.stack((EC_probe_a, EC_probe_b))

    return {
        "autoencoder": autoencoder,
        "EC_laps_by_schedule": (EC_swap_laps, EC_no_swap_laps),
        "schedules": (swap_schedule, no_swap_schedule),
        "EC_probes": EC_probes,
    }


def run_cue_remapping(seed: int, prepared: dict, settings: dict,
                      plasticity_rule: str,) -> tuple[dict[str, np.ndarray], list[dict]]:
    """ Apply one rule to the shared cue-swap/no-swap protocol.

    The scheduled-context probe yields the lap-to-lap curve in panel B.  The
    final two probes of the swap model yield the heatmaps in C/D and per-unit
    stability/modulation values in E.
    """

    memory = settings.get("memory_by_rule", {}).get(
        plasticity_rule, {**settings["memory"], "plasticity_rule": plasticity_rule})
    swap_schedule, no_swap_schedule = prepared["schedules"]

    # Learn each schedule one lap at a time, then probe without learning:
    #
    #       scheduled EC lap → update CA3→CA1 → read-only context A/B probes
    scheduled, final_probes = [], []
    for EC_laps, schedule in zip(prepared["EC_laps_by_schedule"], prepared["schedules"]):
        # Reusing this exact seed across schedules and rules holds the EC→CA3
        # projection fixed. The only designed schedule difference is whether
        # cue identities exchange positions after each ten-lap block.
        model = build_mtl(prepared["autoencoder"], memory, seed_value(seed, 20))
        condition_probes = []
        for EC_lap, assignment in zip(EC_laps, schedule):
            run_mtl(model, EC_lap, learn=True)
            ca1_probes = np.stack([run_mtl(model, EC_probe, learn=False)[1] for EC_probe in prepared["EC_probes"]])
            condition_probes.append(ca1_probes[0 if assignment == [0, 1] else 1])
        scheduled.append(np.stack(condition_probes))
        final_probes.append(ca1_probes)
    scheduled = np.stack(scheduled)
    final_probes = np.stack(final_probes)

    # ------------------------------------------------------------------
    # 4. Convert the probe time series into one number per lap transition.
    #
    # Each transition first compares each CA1 unit's mean-centered spatial
    # tuning curve, then averages across CA1 units.  A low value means the
    # recalled population map changed between the two successive laps.
    # ------------------------------------------------------------------
    transition = np.asarray([
        [tuning_similarity(values[lap + 1], values[lap]).mean() for lap in range(len(values) - 1)]
        for values in scheduled
    ], dtype=np.float32)

    # Final fields come from the swap model, after it has experienced both
    # contexts.  Heatmaps retain position x unit activity; stability and
    # modulation collapse that data to one value per CA1 unit.
    fields = final_probes[0]
    stability = tuning_similarity(fields[0], fields[1])
    modulation = np.mean(np.abs(fields[0] - fields[1]), axis=0)
    context_indices = np.asarray([[0 if assignment == [0, 1] else 1 for assignment in schedule] for schedule in (swap_schedule, no_swap_schedule)])
    cue_changed = context_indices[:, 1:] != context_indices[:, :-1]
    rows = []
    for condition_index, condition in enumerate(("swap", "no_swap")):
        rows.extend(
            {
                "experiment": "cue_schedule",
                "seed": seed,
                "plasticity_rule": plasticity_rule,
                "condition": condition,
                "from_lap": lap + 1,
                "to_lap": lap + 2,
                "cue_changed": bool(cue_changed[condition_index, lap]),
                "tuning_similarity": float(transition[condition_index, lap]),
            }
            for lap in range(transition.shape[1])
        )
    return {"transition": transition, "cue_changed": cue_changed, "probe_ca1": fields, "stability": stability, "modulation": modulation}, rows


def mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Return the across-seed mean and SEM along the first axis """

    values = np.asarray(values, dtype=float)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def save_panel(figure, output: Path, name: str, rule: str, display_rule: str) -> None:
    """ Save editable and raster versions of a rule-specific panel """

    for suffix in ("png", "svg"):
        figure.savefig(output / f"{name}_{rule}.{suffix}", dpi=300, bbox_inches="tight")
        if rule == display_rule:
            figure.savefig(output / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_arrangement_guide(output: Path) -> None:
    """ Draw a simple map from saved plot files to composed Figure 2 panels """

    panels = (
        (0, 2, "A", "Decoder compatibility", ("plot_1_a_base.svg", "plot_1_a_err2.svg")),
        (1, 2, "B", "Full-lap cue transitions", ("plot_1_b_base.svg", "plot_1_b_err2.svg")),
        (0, 1, "C", "Cue-context CA1 maps", ("plot_1_c_err2.svg", "plot_1_d_err2.svg")),
        (1, 1, "D", "Spatial/cue tuning distribution", ("plot_1_e_base.svg", "plot_1_e_err2.svg")),
        (0, 0, "E", "Event-aligned reconfiguration", ("../remapping_dynamics/plot_remap_c_base.svg", "../remapping_dynamics/plot_remap_c_err2.svg")),
        (1, 0, "F", "Population-map similarity", ("../remapping_dynamics/plot_remap_a_err2.svg",)),
    )
    figure, axis = plt.subplots(figsize=(13, 8.5))
    axis.set(xlim=(0, 2), ylim=(0, 3))
    axis.axis("off")
    for column, row, letter, title, files in panels:
        rectangle = plt.Rectangle(
            (column + 0.04, row + 0.06), 0.92, 0.88,
            facecolor="#f7f7f7", edgecolor="#333333", linewidth=1.3,
        )
        axis.add_patch(rectangle)
        axis.text(column + 0.09, row + 0.79, letter, fontsize=18, fontweight="bold", va="top")
        axis.text(column + 0.50, row + 0.78, title, fontsize=11, fontweight="bold", ha="center", va="top")
        axis.text(
            column + 0.50, row + 0.38, "\n".join(files),
            fontsize=8.5, family="monospace", ha="center", va="center",
        )
    axis.text(1.0, 3.02, "Figure 2 assembly map", fontsize=16, fontweight="bold", ha="center", va="bottom")
    for suffix in ("png", "svg"):
        figure.savefig(output / f"figure_2_arrangement.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def representative_fields(fields: np.ndarray, stability: np.ndarray,
                          modulation: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """ Choose the seed closest to median stability/modulation for heatmaps """

    scores = np.column_stack((stability.mean(axis=1), modulation.mean(axis=1)))
    center = np.median(scores, axis=0)
    seed_index = int(np.argmin(np.sum((scores - center) ** 2, axis=1)))
    selected = fields[seed_index]
    order = np.argsort(np.argmax(selected.mean(axis=0), axis=0))
    return selected, order, float(np.quantile(selected, 0.97))


def build_panels(arrays: dict[str, np.ndarray], output: Path, display_rule: str) -> None:
    """ Save Figure-2 panels as individual PNGs for external composition.

    ``plot_1_a.png`` through ``plot_1_e.png`` use ``display_rule``. Matching
    ``_base`` and ``_err2`` files are also saved for rule comparison.
    """

    rules = arrays["plasticity_rules"].tolist()
    conditions = arrays["compatibility_conditions"].tolist()
    schedule_names = arrays["schedule_conditions"].tolist()
    labels = {"aligned": "Aligned",
              "fixed_permutation": "Fixed\npermutation",
              "matched_decoder": "Matched\ndecoder",
              "random_content": "Random\ncontent",
              "no_plasticity": "No\nplasticity"}
    colors = {"swap": "#6a3d9a", "no_swap": "0.40"}

    for rule_index, rule in enumerate(rules):
        rule_label = RULE_LABELS.get(rule, rule)
        values = arrays["compatibility_cosine"][rule_index]
        mean, sem = mean_sem(values)
        figure, axis = plt.subplots(figsize=(4.7, 3.6))
        positions = np.arange(len(conditions))
        for seed_index, seed_values in enumerate(values):
            jitter = np.random.default_rng(7000 + seed_index).uniform(-0.08, 0.08, len(positions))
            axis.scatter(positions + jitter, seed_values, color="0.55",
                         alpha=0.45, s=14, linewidths=0)
        axis.errorbar(positions, mean, yerr=sem, color="black", fmt="o", capsize=3, markersize=5)
        axis.set(xticks=positions, xticklabels=[labels[name] for name in conditions],
                 ylabel="Output–target cosine",
                 ylim=(-0.04, 1.04), title=f"Decoder compatibility ({rule_label})")
        axis.spines[["top", "right"]].set_visible(False)
        save_panel(figure, output, "plot_1_a", rule, display_rule)

        transitions = arrays["transition_similarity"][rule_index]
        changed = arrays["cue_changed"][rule_index]
        figure, axis = plt.subplots(figsize=(5.4, 3.6))
        x = np.arange(2, transitions.shape[-1] + 2)
        for schedule_index, schedule in enumerate(schedule_names):
            mean, sem = mean_sem(transitions[:, schedule_index])
            axis.plot(x, mean, color=colors[schedule], label=schedule.replace("_", " "), linewidth=2)
            axis.fill_between(x, mean - sem, mean + sem, color=colors[schedule], alpha=0.16)
        for position in x[changed[0, 0]]:
            axis.axvline(position, color="#6a3d9a", linestyle="--", linewidth=0.8, alpha=0.45)
        axis.set(xlabel="Probe after lap", ylabel="CA1 tuning similarity",
                 ylim=(-0.05, 1.05), title=f"Cue-schedule transition control ({rule_label})")
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
        save_panel(figure, output, "plot_1_b", rule, display_rule)

        fields, order, vmax = representative_fields(arrays["probe_ca1"][rule_index],
                                                    arrays["spatial_stability"][rule_index],
                                                    arrays["cue_modulation"][rule_index])
        for context_index, name in enumerate(("plot_1_c", "plot_1_d")):
            figure, axis = plt.subplots(figsize=(4.6, 3.6))
            image = axis.imshow(fields[context_index, :, order].T, origin="lower",
                                aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
            figure.colorbar(image, ax=axis, label="CA1 activity")
            axis.set(xlabel="Track position", ylabel="CA1 unit",
                     title=f"Cue context {'A' if context_index == 0 else 'B'} ({rule_label})")
            save_panel(figure, output, name, rule, display_rule)

        figure, axis = plt.subplots(figsize=(4.7, 3.6))
        axis.scatter(arrays["spatial_stability"][rule_index].ravel(),
                     arrays["cue_modulation"][rule_index].ravel(), color="#3d85c6",
                     alpha=0.38, s=10, linewidths=0)
        axis.axvline(0.0, color="0.45", linestyle=":", linewidth=1)
        axis.set(xlabel="Mean-centered spatial stability", ylabel="Cue modulation",
                 title=f"CA1 tuning distribution ({rule_label})")
        axis.spines[["top", "right"]].set_visible(False)
        save_panel(figure, output, "plot_1_e", rule, display_rule)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "results/preprint/figure_2")
    parser.add_argument("--rules", nargs="+", choices=("base", "err2"),
                        default=SETTINGS["plasticity_rules"],
                        help="Plasticity rules to compare; both are paired by default.")
    parser.add_argument("--display-rule", default=None,
                        help="Rule used for unsuffixed plot_1_*.png panels; defaults to err2 when present.")
    parser.add_argument("--quick", action="store_true", help="Run two low-cost seeds for a smoke test.")
    args = parser.parse_args()

    settings = copy.deepcopy(SETTINGS)
    settings["plasticity_rules"] = list(args.rules)
    apply_evolved_parameters(settings, ROOT_DIR)
    if args.quick:
        settings["seeds"] = settings["seeds"][:2]
        settings["autoencoder"]["epochs"] = 8
        settings["compatibility"]["epochs"] = 8
        settings["compatibility"]["training_patterns"] = 128
        settings["compatibility"]["validation_patterns"] = 32

    # ------------------------------------------------------------------
    # 5. Repeat the paired protocol across 20 independent seeds and save raw
    # arrays.  Plotting code can later choose its own error-bar style without
    # rerunning the scientific simulation.
    #
    # Replicates, not individual units or masks, are the independent samples.
    # ------------------------------------------------------------------
    compatibility, transitions, changes, fields, stability, modulation, rows = [], [], [], [], [], [], []
    for index, seed in enumerate(settings["seeds"], start=1):
        print(f"Figure 2 seed {seed} ({index}/{len(settings['seeds'])})", flush=True)
        compatibility_setup = prepare_compatibility(seed, settings)
        cue_setup = prepare_cue_remapping(seed, settings)
        rule_compatibility, rule_transitions, rule_changes = [], [], []
        rule_fields, rule_stability, rule_modulation = [], [], []
        for plasticity_rule in settings["plasticity_rules"]:
            cosine, seed_rows = run_compatibility(seed, compatibility_setup,
                                                  settings, plasticity_rule)
            remapping, remapping_rows = run_cue_remapping(seed, cue_setup,
                                                          settings, plasticity_rule)
            rule_compatibility.append(cosine.mean(axis=1))
            rule_transitions.append(remapping["transition"])
            rule_changes.append(remapping["cue_changed"])
            rule_fields.append(remapping["probe_ca1"])
            rule_stability.append(remapping["stability"])
            rule_modulation.append(remapping["modulation"])
            rows.extend(seed_rows)
            rows.extend(remapping_rows)
        compatibility.append(np.stack(rule_compatibility))
        transitions.append(np.stack(rule_transitions))
        changes.append(np.stack(rule_changes))
        fields.append(np.stack(rule_fields))
        stability.append(np.stack(rule_stability))
        modulation.append(np.stack(rule_modulation))

    arrays = {
        "root_seeds": np.asarray(settings["seeds"]),
        "plasticity_rules": np.asarray(settings["plasticity_rules"]),
        "compatibility_conditions": np.asarray(COMPATIBILITY_CONDITIONS),
        "compatibility_cosine": np.stack(compatibility).swapaxes(0, 1),
        "schedule_conditions": np.asarray(("swap", "no_swap")),
        "transition_similarity": np.stack(transitions).swapaxes(0, 1),
        "cue_changed": np.stack(changes).swapaxes(0, 1),
        "probe_ca1": np.stack(fields).swapaxes(0, 1),
        "spatial_stability": np.stack(stability).swapaxes(0, 1),
        "cue_modulation": np.stack(modulation).swapaxes(0, 1),
    }
    display_rule = args.display_rule or ("err2" if "err2" in settings["plasticity_rules"] else settings["plasticity_rules"][0])
    if display_rule not in settings["plasticity_rules"]:
        raise ValueError("--display-rule must be one of the selected --rules")
    artifact = write_artifact(args.output, settings, arrays, rows)
    build_panels(arrays, artifact, display_rule)
    build_arrangement_guide(artifact)
    print(artifact)


if __name__ == "__main__":
    main()
