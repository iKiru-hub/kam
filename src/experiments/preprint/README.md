# Preprint simulations

This folder contains the small, final simulation layer for the preprint.  It
does not replace the exploratory scripts in `src/experiments/`.

Each runner creates a new artifact folder containing the resolved
configuration, compressed raw arrays, a tidy CSV table, a report, and a Git
revision manifest.  Runners refuse to overwrite a nonempty folder.

Run the simulations from the repository root:

```sh
PYTHONPATH=src python3 -m experiments.preprint.compatibility \
  --config src/experiments/preprint/configs/final_compatibility.json \
  --output results/preprint/v1/compatibility

PYTHONPATH=src python3 -m experiments.preprint.cue_remapping \
  --config src/experiments/preprint/configs/final_cue_remapping.json \
  --output results/preprint/v1/cue_remapping

PYTHONPATH=src python3 -m experiments.preprint.completion \
  --config src/experiments/preprint/configs/final_completion.json \
  --output results/preprint/v1/completion

PYTHONPATH=src python3 -m experiments.preprint.plasticity_ablation \
  --config src/experiments/preprint/configs/final_plasticity_ablation.json \
  --output results/preprint/v1/plasticity_ablation
```

Build figures only from saved artifacts:

```sh
PYTHONPATH=src python3 -m experiments.preprint.figures.figure_2_compatibility \
  --artifact results/preprint/v1/compatibility \
  --output article/figures/figure_2_compatibility.png
```

`compatibility.py` is the primary result.  It compares aligned instructions,
fixed coordinate mismatch, matched decoder coordinates, random content, and
no plasticity under paired seed-specific inputs.

`cue_remapping.py` trains on a fixed cue-swap schedule, freezes learning, and
measures CA1 spatial stability and cue modulation under two probe contexts.

`completion.py` stores clean laps, freezes learning, corrupts probes with
independent masks, and compares normal, shuffled, dense, and identity CA3 key
maps.  It reports both output recovery and clean-corrupted CA3 key overlap.

`plasticity_ablation.py` compares the base instruction-gated update with the
bounded error-driven update (`err2`) using the same autoencoder, CA3 key map,
cue-track schedule, learning rate, and paired corruption masks.  It is a
fixed-parameter control, not a per-rule optimization study.
