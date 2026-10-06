# Readability and evidence revision (28 September 2026)

- Rewrote the abstract (approximately 230 words), Introduction, and Discussion
  to distinguish readout compatibility from information preservation, latent
  stability, recurrent completion, and a biological mechanism for compatibility.
- Added explicit positioning relative to Schapiro et al. (2017), Chandra et al.
  (2025), and Rule and O'Leary (2022), and incorporated Dorian et al. (2026).
  Checked these claims against primary publications or author-maintained sources.
  Corrected Chandra's final volume and page range.
- Clarified the remapping confound: scheduled-context probes change the query
  as well as the learning history. The two boundary summaries use separate
  random substreams, not identical trajectories.
- Corrected the search's nominal 85% dropout to its implemented 21/25 (84%),
  and specified clipping of the largest CA3 activity multiplier to 49.
- Documented the different two-cue and ten-cue template sparsities, the nominal
  independent-guess chance level, and permutation equivariance during learning.
- Found stale embedded sensitivity/load panels in the legacy Figure3.svg.
  Replaced them with current saved results as the new Figure 4; Figure 3 now
  contains degradation panels A-D only. Updated all manuscript references.
  Added a composition script and improved cell/legend legibility.
- Rechecked reported load means/SEMs and sensitivity extrema against saved
  numerical outputs. No simulations or scientific data were changed.

## Reproduction

Regenerate the cue-pair panel with plot_preprint_capacity in
src/experiments/preprint_multiple_cues.py, using the existing
results/preprint/multiple_cues/condition_data.csv, then run:

    MPLCONFIGDIR=/tmp/kam-mpl python3 src/experiments/compose_preprint_figures.py
    latexmk -cd -pdf -synctex=1 -interaction=nonstopmode -halt-on-error article/maintex/main.tex

The composition script retains only A-D from the legacy Figure3.svg and
recreates Figure 4 from current sensitivity arrays and the current load panel.
The legacy composite itself must not be exported as the manuscript figure.

## Items that prose revision cannot resolve

- Supply the verified author list/affiliations and a permanent code/data archive.
- Separate fixed-context learning effects from fixed-weight context switches
  before attributing the transition discontinuity specifically to plasticity.
- Add sparsity/dimensionality-matched key controls for claims about sparse wiring.
- Control total writes, vocabulary exposure, and template sparsity for stronger
  capacity comparisons; use an independent query/target task for broader
  cue-to-episode retrieval claims.
