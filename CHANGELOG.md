# Changelog

All notable changes to Taxonova are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions use calendar
versioning, `YY.MM.MICRO`: the year and month of the release, then a counter
for further releases in the same month. Model files carry their own version
(see [docs/models.md](docs/models.md)).

## [Unreleased]

## [26.10.1] - 2026-10-30

Usability release. Classification output is unchanged: for the same model and
thresholds, every table is byte-identical to 26.10.0.

### Added

- `taxonova classify INPUT`: a shorter command. The model is chosen with
  `--model` (default `v4`) and downloaded the first time it is used; the output
  goes to `<input name>.taxonova.tsv` in the current directory, with the run
  log beside it (`--no-log` to skip). Short options `-m`, `-o`, `-c`
  (`--confidence`), `-t` (`--threads`) and `--<rank>-threshold`.
- `taxonova models` lists the models and which ones are downloaded;
  `taxonova models download <id ...|all>` downloads them.
- `taxonova --version`. Running `taxonova` with no arguments prints the help.
- Short model ids: `v4`, `v3v4`, `v5v7`, `16s` (whole 16S, V1-V9) and `operon`
  (16S-ITS-23S). The model files have readable names, for example
  `taxonova-V4-greengenes2-26.10.pt`; they are the same files as before (same
  checksums), and the earlier keys and file names still work (see
  [docs/models.md](docs/models.md#earlier-names)).
- `taxonova.models`: a small, stable Python interface to the model registry
  (`list_models`, `get_model`, `ensure_model`, `models_dir`), used by the web
  app and TaxoBench.
- `taxonova.classify(...)` returns the classification as a pandas DataFrame
  with the same columns as the command line.
- Models are downloaded from GitHub and, if that fails, from a mirror on
  Hugging Face (`claessonlab/taxonova-models`). Every download is checked
  against the published sha256 (and md5); the operon model now has a sha256.
  Several jobs that start at the same time can share one models directory:
  each downloads to its own temporary file, and the first verified copy is
  used.
- FASTA input may be gzip-compressed (`.gz`).
- RNA input is accepted: `U` is read as `T`, so the RNA and DNA spellings of a
  sequence give the same result. Before, every k-mer containing `U` was
  ignored.
- Documentation in `docs/`, a citation file (`CITATION.cff`), issue templates,
  continuous integration on Python 3.9 to 3.12 and a release workflow.

### Changed

- `classification --model_input` also accepts a model id, and then downloads
  the model if needed. Paths and file names behave as before.
- `download-models --models` also accepts the new ids.
- The download progress line is refreshed at most twice a second, and only on
  an interactive terminal; before, a log or batch job received a line for every
  8 KB.
- The torch warning about `weights_only` is no longer printed when a model is
  loaded, and the help text of `calibrate` and `fit-ood` is plainer.
- The source distribution contains only the files listed in `MANIFEST.in`;
  before, untracked files under `tests/` could be included.
- Python 3.9 or later is required (3.8 is no longer tested).
- The README is shorter. Benchmark figures are in the paper, and the history
  of the Greengenes2 checkpoints is below.

## [26.10.0] - 2026-10-29

**Releases 26.9.0 to 26.9.2 applied no confidence thresholds to the four
Greengenes2 checkpoints (V4, V3-V4, V5-V7, whole 16S).** For about 98 to 100% of
reads `--confidence_threshold` and the per-rank thresholds had no effect, and
Taxonova named a species for almost every read. The same code path is present in
1.0.0 and 26.6.0, which used the same checkpoints. The RRN checkpoint carries no
OOD configuration and was not affected.

This release fixes the code and replaces the four Greengenes2 checkpoints.
Results produced with the Greengenes2 checkpoints under any earlier release
differ from 26.10.0 output; rerun them.

### What was wrong

1. The Greengenes2 checkpoints carry an out-of-distribution (OOD) detector
   (`ood_config`). A read flagged OOD was gated with the checkpoint's
   `relaxed_thresholds`, which are 0.0 at every rank, instead of the requested
   thresholds.
2. `fit-ood --id_quantile 0.99 --score_method energy_phylum` placed tau at the
   1st percentile of in-distribution energies, so about 99% of in-distribution
   reads were flagged OOD. The phylum-MSP path read `--id_quantile` the other way
   round (default 0.01 meant "flag 1%"). Together with (1), almost every read was
   classified with every threshold at 0.0.
3. Fixing (1) and (2) alone does not give usable output from the v1.0.0
   checkpoints: their confidences were rescaled by tau-normalisation, so most
   species confidences sit below the documented thresholds (whole 16S: median
   species abstention 99% at 0.5/0.7 on the Taxobench mock communities).

### What changed

- Code: a read flagged OOD is gated at the per-rank maximum of the requested
  threshold and the checkpoint's OOD threshold, so an OOD flag can no longer
  relax a call. `--id_quantile` now means the fraction of held-out
  in-distribution samples kept as in-distribution, for both score directions
  (default 0.99, about 1% flagged; values outside (0.5, 1) are rejected; new
  `ood_config` entries carry `id_quantile_convention: "id_coverage"`). The
  classification log reports the share of inputs flagged OOD and warns above 20%
  or when tau was fitted under the old convention.
- New Greengenes2 checkpoints, attached to the v26.10.0 GitHub release under new
  file names (`gg2_<region>_mrgroups_model_26.10.pt`), so files cached by 26.9.x
  are never reused by mistake:
  - V1-V9 and V5-V7: the released networks with tau-normalisation (tau 0.5)
    undone exactly (re-applying it reproduces the released weights to 1.2e-7).
  - V4 and V3-V4: tau-normalisation at tau 1.0 cannot be undone, so these were
    retrained on the same reference with the same recipe, without
    tau-normalisation and without the OOD gate (seeds 1005 and 1006, chosen by
    lowest final training loss, a rule fixed before any evaluation).
  - The OOD gate is dropped: none of the four checkpoints carries `ood_config`,
    as for RRN. On leave-out tests a stricter OOD gate lowered the rate at which
    novel species were named by at most 0.5 percentage points.
  - The RRN checkpoint is unchanged (v1.0.0 asset).
- Default thresholds are unchanged: 0.5 for domain to family, 0.7 for genus and
  0.8 for species, scaled by `--confidence_threshold`.
- `taxonova download-models` verifies a cached file against the published md5
  and sha256 before using it; a file that does not match is reported and
  downloaded again. Downloads are written to a temporary file and moved into
  place only after the checksums pass.
- `taxonova download-models` (and `taxonova-setup`) now store models in
  `TAXONOVA_MODELS_DIR` when it is set, the directory `classification` already
  searched; before, they always used `~/.taxonova/models/`.
- Classifying with a v1.0.0 Greengenes2 checkpoint (file name and md5 match)
  prints a warning that names its replacement.
- Tests: `tests/test_ood_gating.py` (gating cases, quantile convention, regression
  fixture) and `tests/test_release_26_10.py` (downloader checksums and cache
  handling, threshold 0.9 known-answer gating, and, when the 26.10 checkpoints
  are available, a regression of real classifications at 0.9).

### Expected changes in output

More reads stop above species, as the documentation describes. Median
species-level F-measure over the 13 Taxobench mock communities per region
(Taxobench evaluator v1 rules, corrected gutmock V5-V7 ground truth), with median
species abstention:

| region | 26.9.2 (no thresholds in effect) | v1.0.0 checkpoints, fixed code, 0.5/0.7 | 26.10.0, 0.5/0.7 | 26.10.0, default (0.8) |
| --- | --- | --- | --- | --- |
| whole 16S (V1-V9) | 0.962, 0% | 0.028, 99% | 0.961, 4.1% | 0.945, 7.2% |
| V5-V7 | 0.953, 0% | 0.672, 49% | 0.940, 4.3% | 0.940, 4.3% |
| V4 | 0.988, 0% | 0.636, 50% | 0.957, 5.3% | 0.945, 8.7% |
| V3-V4 | 0.863, 0% | 0.724, 40% | 0.836, 19.5% | 0.797, 26.7% |

"0.5/0.7" is 0.5 for domain to genus and 0.7 for species, the thresholds of the
published benchmark runs. Species precision rises or holds and recall falls; at
V3-V4 about one read in five now stops above species at 0.5/0.7. On the
vaginal mock community at V5-V7, 26.10.0 names 7 of 12 sequences at species and
abstains on 5 (F 0.737). On leave-out tests, where the query species is absent
from the reference, the new checkpoints name a (necessarily wrong) species for
more novel queries than the tau-normalised ones: 0.14 at whole 16S and 0.18 at
V5-V7 at 0.5/0.7, against 0.02 and 0.00 before; at the default 0.8 these are
0.11 and 0.14.

## [26.9.2] - 2026-09-18

Documentation and metadata only, no code change. The README, the package summary
and the command-line help no longer describe Taxonova as a hierarchical classifier:
classifiers trained at each rank alone reached similar species-level accuracy, so
the rank ordering is not presented as the source of its accuracy. The README is
shorter and points to the web server first. QUICKSTART now gives the species
threshold default as 0.8, as the software has used since 26.9.0, and the model
tables name the whole-16S region and the GROND reference.

## [26.9.1] - 2026-09-06

Metadata only, no code change. Corrects the author's surname to Kakabra, which
26.9.0 and earlier spelt "Kaka Bra".

## [26.9.0] - 2026-09-06

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

## [26.6.0] - 2026-06-06

Not previously recorded here.

- Ported and modernised the pretrained-model downloader.
- PEP 621 packaging: metadata moved into `pyproject.toml`.


## [1.0.0] - 2026-06-06

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
