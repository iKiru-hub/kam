# Preprint experiment guide

This preprint has two main numerical experiments, one optional remapping-
dynamics extension, and one conceptual figure.
The scripts here generate data only; figure layout, fonts, panel labels, and
captions belong in the article figure-composition layer.

## The minimal code path

```text
preprint_figure_2.py / preprint_figure_3.py / preprint_remapping_dynamics.py
                |
                v
        preprint_common.py       experiment bookkeeping only
                |
      +---------+----------+
      |                    |
      v                    v
 core.datagen          core.ae_tools
 core.models           core.mtl_tools
```

`core.datagen` makes the inputs. `core.models` contains the EC--CA1 encoder,
the CA3 retrieval key, and plastic CA3--CA1 weights. `core.ae_tools` pretrains
the encoder, and `core.mtl_tools.run_sequence` runs an ordered lap without
resetting the associative memory. `preprint_common.py` is deliberately small:
it fixes seeds, defines the exact cue schedule, and saves standard artifacts.

## Paired plasticity-rule comparison

Both figure-data scripts run `base` and `err2` by default. Within a root seed,
the rules share EC inputs, the pretrained EC--CA1--EC encoder, fixed decoder,
CA3 wiring, storage order, and (for Figure 3) degradation masks. Only the
CA3--CA1 update equation changes.

Every numerical array starts with a plasticity-rule dimension followed by the
root-seed dimension. You can run one rule for debugging, for example
`pp preprint_figure_3.py --rules base`, but paper comparisons should use the
default paired two-rule run.

After both figure-data experiments finish, run:

```bash
pp preprint_plasticity_comparison.py
```

This writes `plasticity_comparison.png`, `plasticity_comparison.svg`,
`seed_level.csv`, and `summary.csv` to
`results/preprint/plasticity_comparison/`. It is the concise supplementary
overview; the full Figure 2 and 3 arrays retain every curve and probe.

## Figure 2: compatibility and cue-dependent CA1 reconfiguration

Run it with:

```bash
pp preprint_figure_2.py
```

The script contains two linked but separate protocols.

### Panel A: coordinate compatibility

The encoder is first pretrained on binary sparse EC patterns. Its EC--CA1
encoder and CA1--EC decoder are then frozen. Twenty-eight new patterns are
stored by associating each sparse CA3 key with an instructed CA1 target under
both rules.

The five conditions hold the stored EC patterns, CA3 keys, encoder, decoder,
and presentation order fixed. They differ only in the meaning assigned to the
CA1 target:

| Condition | Manipulation | Purpose |
| --- | --- | --- |
| `aligned` | Use the encoder-derived CA1 target unchanged. | Reference condition. |
| `fixed_permutation` | Permute every CA1 target coordinate; leave decoder fixed. | Test coordinate mismatch. |
| `matched_decoder` | Apply the same permutation to the decoder columns. | Algebraic restoration control. |
| `random_content` | Store another pattern's CA1 target. | Incorrect-association control. |
| `no_plasticity` | Do not write CA3--CA1 weights. | Unwritten-memory control. |

The metric is output-target cosine for each of 28 stored patterns, averaged
within each seed. The claim is narrowly about *decoder-compatible coordinates*,
not that CA1 activity in general carries insufficient information.

### Panels B--E: cue swap and matched control

An input lap has 25 MEC-like spatial units and 25 LEC-like cue units. Two cue
positions stay fixed at positions 10 and 30. The cue identities are either
kept as `[0, 1]` on every lap or swapped to `[1, 0]` every ten laps.

For each seed, both schedules and both rules use the same MEC trajectory,
pretrained encoder, exact EC--CA3 projection, and plasticity settings. A model
learns one lap, plasticity is paused, and it is probed in both cue contexts.
This lets the analysis ask whether tuning changes occur at the schedule
transition, rather than being an artifact of continued learning during
readout or a different retrieval-key projection.

| Array | Shape in a full run | Figure use |
| --- | --- | --- |
| `transition_similarity` | rules x seeds x 2 schedules x 39 transitions | B: mean tuning similarity after each lap transition. |
| `cue_changed` | rules x seeds x 2 x 39 | Identifies the swap transitions in B. |
| `probe_ca1` | rules x seeds x 2 contexts x 50 positions x 50 units | C/D: same CA1 units in the two cue contexts. |
| `spatial_stability` | rules x seeds x 50 units | E x-axis: mean-centered spatial-tuning similarity. |
| `cue_modulation` | rules x seeds x 50 units | E y-axis: mean absolute cross-context change. |

The heatmap realization should be chosen by a stated, reproducible rule (the
existing manuscript uses the realization nearest the median stability and
modulation). The individual units in panel E are descriptive samples nested
within seeds; seed means are the independent quantities for uncertainty
intervals.

## Figure 3: selective MEC/LEC degradation

Run it with:

```bash
pp preprint_figure_3.py
```

Each replicate pretrains an encoder, stores 20 *clean* cue-track laps through
each selected CA3--CA1 rule, and then freezes plasticity. It probes the last
clean lap under selective input dropout. The mask deletes a fixed subset of
input units for the whole lap, and a new mask is drawn for the next probe.

This design distinguishes two questions:

| Corrupted input half | Main readout | Intact-modality control | Figure panel |
| --- | --- | --- | --- |
| LEC cue input | Cue identity at the two cue locations | MEC output-target cosine | A/B |
| MEC spatial input | Nearest-position accuracy from MEC output | Cue identity from LEC output | C/D |

The experiment evaluates five dropped fractions (0, .25, .50, .75, .90), 12
masks per fraction, and 20 seeds. First average the 12 masks within a seed;
then use the 20 seed-level values to compute a mean and SEM/CI.

The `normal` key condition uses the model's sparse EC-to-CA3 wiring. The
`dense` control assigns every CA3 unit the same average over EC coordinates.
It is intentionally a degenerate key: it tests the necessity of
item-selective retrieval keys, not the more limited question of whether one
sparsity level is better than another.

| Array | Shape in a full run | Figure panel |
| --- | --- | --- |
| `lec_cue_accuracy` | rules x seeds x key modes x fractions x masks | A |
| `lec_mec_cosine` | rules x seeds x key modes x fractions x masks | B |
| `mec_position_accuracy` | rules x seeds x key modes x fractions x masks | C |
| `mec_cue_accuracy` | rules x seeds x key modes x fractions x masks | D |

The appropriate conclusion is selective, graceful recall under partial input
loss. It should not be phrased as recurrent pattern completion because this
model has no recurrent CA3 completion stage and cannot infer a modality after
the available cue information has been eliminated.

## Optional extension: remapping dynamics after cue exchange

Run it with:

```bash
pp preprint_remapping_dynamics.py
```

This experiment follows CA1 after repeated exchanges of the two cue
identities. It uses the same 25 MEC-like spatial units, 25 LEC-like cue units,
pretrained encoder, and CA3--CA1 plasticity implementation as the main
experiments. Each full run contains 40 laps divided into four ten-lap blocks.
The cue-exchange schedule swaps cue identities at every block boundary; the
matched no-exchange schedule presents the same context throughout.

Within each seed, the two schedules and the two plasticity rules share the
pretrained encoder and the exact EC--CA3 projection. Plasticity is enabled for
one training lap, then paused while the model is probed with the complete
scheduled context. This train-then-probe cycle yields a CA1 population map
after every lap without contaminating the readout with an additional update.

The event-aligned analysis defines lap 0 as the first lap with exchanged cues.
The final pre-exchange map is the old-map reference. The last lap of the new
ten-lap block is the adapted-map reference. Similarity to the latter is a
retrospective description of convergence across the block, not a causal
prediction of a future representation.

| Output | Question answered | Suggested use |
| --- | --- | --- |
| `plot_remap_a` | Do CA1 population states form stable blocks separated by cue exchanges? | Population representational-similarity matrix. |
| `plot_remap_b` | Does the current map move away from the old map and toward the adapted map? | Event-aligned old/adapted-map similarity. |
| `plot_remap_c` | Is reconfiguration concentrated at the cue-exchange lap rather than ordinary repeated exposure? | Cue-exchange versus no-exchange step similarity. |
| `plot_remap_d` | How broadly is the effect distributed across CA1 units? | Unit-level endpoint-similarity distribution. |

Each plot is written as PNG and SVG. As in the other scripts, rule-specific
files retain the internal `_base` or `_err2` suffix, while labels inside the
plots use “instructive-driven” and “error-driven.” The unsuffixed copy uses the
display rule selected on the command line. Numerical outputs are stored in
`results/preprint/remapping_dynamics/`.

Unit-level curves and distributions are descriptive because units are nested
within simulations. Statistical uncertainty should be computed from root-seed
replicates, after averaging repeated swap events within each seed.

## Output and checks

The scripts write these files to their respective directory under
`results/preprint/`:

- `arrays.npz`: all numerical arrays used by plot code;
- `config.json`: exact parameters and random seeds;
- `source_data.csv`: tidy per-memory, per-lap, or per-mask values.

Use `--quick` first. Across the scripts this reduces the seed count,
pretraining, and any repeated corruption samples; it only checks that the
pipeline works. Never use its numerical values in the paper.
