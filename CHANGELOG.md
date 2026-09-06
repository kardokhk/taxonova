# Changelog

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
