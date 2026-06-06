# Changelog

## 1.0.0

The first public release, and the canonical version that matches the manuscript.

This is the full Taxonova system as described in the paper:

- Hierarchical domain-to-species cascade with cross-level attention, so each rank
  is predicted in the context of its parents rather than as seven independent
  classifiers.
- Calibrated abstention: the model returns a genus-level call instead of a wrong
  species call when a region cannot resolve the species, controlled by per-rank
  confidence thresholds.
- Tau-norm confidence rescaling, baked into the released checkpoints.
- Pretrained models for five regions: V4, V3-V4, V5-V7, and V1-V9 (Greengenes2),
  plus the full rRNA operon RRN (GTDB). These are attached to the GitHub Release.
- CLI with `training`, `classification`, `calibrate`, and `fit-ood` modes, runnable
  on a single CPU core.
