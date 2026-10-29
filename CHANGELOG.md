# Changelog

## Unreleased (branch fix/ood-threshold-gating, not released)

Fixes a defect in releases 26.9.0 to 26.9.2 that switched off the confidence
thresholds for the Greengenes2 checkpoints (V4, V3-V4, V5-V7, whole 16S).

### What was wrong

1. `fit-ood --id_quantile 0.99 --score_method energy_phylum` placed tau at the
   1st percentile of in-distribution energies, so about 99% of in-distribution
   reads were flagged out-of-distribution (OOD). The phylum-MSP path read
   `--id_quantile` the other way round (default 0.01 meant "flag 1%").
2. A read flagged OOD was gated with the checkpoint's `relaxed_thresholds`,
   which are 0.0 at every rank, instead of the requested thresholds. With
   both defects, `--confidence_threshold` had no effect for about 99% of reads.

### What changed

- `--id_quantile` now means the fraction of held-out in-distribution samples
  kept as in-distribution, for both score directions: 0.99 flags about 1%.
  The default is 0.99 (was 0.01) and values outside (0.5, 1) are rejected.
  New `ood_config` entries carry `id_quantile_convention: "id_coverage"`.
- A read flagged OOD is gated at the per-rank maximum of the requested
  threshold and the checkpoint's OOD threshold. An OOD flag can make a call
  more conservative or leave it unchanged; it can no longer relax it. With the
  shipped OOD thresholds of 0.0, output equals a run without `ood_config`.
- The classification log reports the number and share of inputs flagged OOD,
  warns when the share exceeds 20%, and warns when the checkpoint's tau was
  fitted under the old convention.
- Tests: `tests/test_ood_gating.py` (known-answer gating cases, the quantile
  convention for both score directions, a regression fixture under
  `tests/fixtures/`).

### Effect on results

Output from the Greengenes2 checkpoints changes at every threshold: reads whose
confidence is below the requested gate are now reported as `Unclassified`, as
the documentation describes. Abstention rises markedly, most at whole 16S,
where the checkpoints' species confidences are low (tau-normalisation). Fixing
the gate does not recalibrate those confidences; that needs new checkpoints.

## 26.9.2

Documentation and metadata only, no code change. The README, the package summary
and the command-line help no longer describe Taxonova as a hierarchical classifier:
classifiers trained at each rank alone reached similar species-level accuracy, so
the rank ordering is not presented as the source of its accuracy. The README is
shorter and points to the web server first. QUICKSTART now gives the species
threshold default as 0.8, as the software has used since 26.9.0, and the model
tables name the whole-16S region and the GROND reference.

## 26.9.1

Metadata only, no code change. Corrects the author's surname to Kakabra, which
26.9.0 and earlier spelt "Kaka Bra".

## 26.9.0

Two behaviour changes. Read the first one before rerunning an existing analysis.

### The per-rank confidence gates now scale with `--confidence_threshold`

The recommended profile is 0.5 for domain through family, 0.7 for genus and 0.8
for species. Asking for a single species confidence now scales the whole
profile in that ratio, so one number sets the operating point at every rank.

Previously the parent gates were pinned at a flat 0.5 whatever was asked for.
`--confidence_threshold 0.8` still accepted a genus label at 0.5, and because
abstention propagates downward, a parent failing that fixed gate discarded an
already-computed species call.

`--confidence_threshold` also now **defaults to 0.8**, not 0.5, and its help
text says so instead of claiming 0.7 while defaulting to 0.5.

**This changes classification output at default settings.** To reproduce
results from 26.6.0 or earlier, state the old gates explicitly:

    taxonova classification --input seqs.fa --model_input model.pt \
        --output out.tsv --confidence_threshold 0.7 \
        --domain_threshold 0.5 --phylum_threshold 0.5 --class_threshold 0.5 \
        --order_threshold 0.5 --family_threshold 0.5 --genus_threshold 0.5

An explicit `--<level>_threshold` overrides any rank, so a pinned profile is
unaffected by this default. The benchmark in the Taxobench repository states
these gates explicitly for exactly that reason.

The profile now lives in `taxonova.core.thresholds` (`GRADED_PROFILE`,
`GRADED_SPECIES`, `scaled_profile`), which the command-line tool and the hosted
web application both read, so the two entry points cannot drift apart. They
had: the web page defaulted to 0.8 while the command-line tool defaulted to
0.5, and the same sequence at the same nominal confidence was classified to a
different depth depending on which one you used. The module is stdlib-only, so
`--help` still works where the deep-learning stack is missing or broken.

### The weight EMA is bias-corrected

The exponential moving average of the weights retained its zero
initialisation, so early-training averages were biased toward zero.

The released checkpoints were trained with the pre-fix code and are **not**
superseded. At Greengenes2 scale there are thousands of optimiser steps per
epoch, so the retained initialisation is effectively zero and the averages are
unaffected. The fix matters when training on a small reference, which is the
case a user building a model from their own curated database will hit.

## 26.6.0

Not previously recorded here.

- Ported and modernised the pretrained-model downloader.
- PEP 621 packaging: metadata moved into `pyproject.toml`.


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
