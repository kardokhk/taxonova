# Changelog

All notable changes to Taxonova are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions use calendar
versioning, `YY.MM.MICRO`: the year and month of the release, then a counter
for further releases in the same month. Model files carry their own version
(see [docs/models.md](docs/models.md)).

## [Unreleased]

## [26.11.1] - 2026-11-04

**V4-V5 models: new files with a species gate.** `v4v5`, `v4v5-silva` and
`v4v5-rdp` now download new files (`taxonova-V4V5-<reference>-26.11.1.pt`)
that hold the same networks, cut-offs and group-label handling as 26.11.0,
together with a species abstention gate. Some species calls are now
withheld, so results differ from 26.11.0 at the species rank only; every
other rank is unchanged. The 26.11 files remain available as
`v4v5-26.11`, `v4v5-silva-26.11` and `v4v5-rdp-26.11`. The four Greengenes2
models for V4, V3-V4, V5-V7 and the whole gene and the operon model are
unchanged.

### Added

- **Species gate for the V4-V5 models.** As in the four Greengenes2 models,
  the species call is withheld when the read's species feature is far from
  the training records of the predicted species (Mahalanobis distance to the
  species centroid). The cut of each model is listed in
  [docs/models.md](docs/models.md#operating-points). It was fitted on
  validation reads as the strictest cut that keeps species F1 within 0.0075
  of the ungated value (0.005 for RDP, where 0.0075 cost more than 0.01), and
  is the median over three training seeds. On validation reads, cross-fitted,
  the gate changed species F1 of the released network by -0.008 (95% CI
  -0.014 to -0.002) for Greengenes2, -0.006 (-0.012 to -0.0003) for SILVA and
  -0.002 (-0.012 to 0.006) for RDP. On synthetic chimeras of two genera, the
  share of reads with a species call fell from 0.38 to 0.11 (Greengenes2),
  0.21 to 0.10 (SILVA) and 0.18 to 0.06 (RDP); on reads with 3% random
  substitutions, from 0.68 to 0.40, 0.38 to 0.26 and 0.33 to 0.19.

### Changed

- **V4-V5 Greengenes2 model, label rules.** `v4v5` now has the genus-token
  check on, with the table of renamed genera, like the other Greengenes2
  models. This only matters with `--label-rules`; `--genus-token-check off`
  restores the 26.11.0 setting. On validation reads with `--label-rules`,
  species F1 changed by -0.0005 (95% CI -0.0016 to 0.0004); 5 of 2,909 reads
  changed.

### Fixed

- `taxonova.labels.is_group` read a bracketed single name such as the SILVA
  genus `[Eubacterium]` as a one-member group, so that it shared its member
  with the plain genus `Eubacterium`. A group label now needs two or more
  members separated by `|`. This affects `--label-rules` and the
  `taxonova.labels` functions; the default `taxonova classify` output does
  not use them.

### Evaluation of the 26.11.0 files (descriptive)

The 26.11.0 files were scored at their default settings after the release,
on benchmark data that had already been read during development. These
numbers describe the files; nothing was chosen or tuned on them. Species F1
(95% CI) on the real-read pools of the TaxoBench T2 tier, 26.11.0 file
against the 26.10 file at its default, with the paired difference:

- V4 (`v4`): 0.833 (0.696 to 0.932) against 0.756; +0.077 (0.027 to 0.149).
- V3-V4 (`v3v4`): 0.718 (0.511 to 0.872) against 0.625; +0.093 (0.023 to
  0.207).
- Whole gene (`16s`): 0.866 (0.767 to 0.945) against 0.833; +0.033 (0.007 to
  0.074).
- V4-V5 (`v4v5`, without the 26.11.1 gate), against the 26.10 whole-gene
  model that V4-V5 users had before: 0.756 (0.608 to 0.882) against 0.310;
  +0.446 (0.229 to 0.670).

The pools hold 27 to 117 species clusters, so the intervals are wide. T2 has
no V5-V7 reads. On in-silico V4-V5 amplicons of genomes new in GTDB R226,
`v4v5` (26.11.0) had species F1 0.739 (0.675 to 0.799), against 0.202 for
the 26.10 whole-gene model; +0.538 (0.453 to 0.620).

### Known limitations

- The V4-V5 species gate was fitted after the benchmark results had been
  read, on validation data only, and has not been tested on unread data.
- **RDP V4-V5 gate.** Most RDP species have one training record, so the cut
  lies above every training distance and the gate acts only on reads far
  from any training record. On the validation reads (the data it was fitted
  on), `taxonova classify` with the 26.11.1 file had species F1 0.016 lower
  (95% CI 0.002 to 0.034) than with the 26.11 file, above the 0.01 target.
- The limitations of the V4-V5 models listed for 26.11.0 (species-group
  confirmation not met for SILVA; no SILVA models for the other regions)
  still apply. The two items on the species gate and on the Greengenes2
  label rules are resolved.

### Compatibility

- New downloads: 565 MB (`v4v5`), 509 MB (`v4v5-silva`) and 223 MB
  (`v4v5-rdp`), once. The new files load with `torch.load(...,
  weights_only=True)` under torch 2.6 and later, as the 26.11 files do.

## [26.11.0] - 2026-11-04

**Default output changes for the four Greengenes2 models.** `v4`, `v3v4`,
`v5v7` and `16s` now download new files (`taxonova-<REGION>-greengenes2-26.11.pt`)
that hold the same networks as 26.10 together with a calibrated operating point:
per-rank cut-offs and a species abstention gate. When you give no threshold,
these cut-offs replace the 0.5 / 0.7 / 0.8 defaults, so results differ from
26.10 runs at the default settings. The 26.10 files remain available as
`v4-26.10`, `v3v4-26.10`, `v5v7-26.10` and `16s-26.10`.

**Python default changes too.** `taxonova.classify()` now uses the same
thresholds as the command line: without `confidence=` (and without any
per-rank threshold) it uses the model's operating point. See "Changed" for
how to keep the earlier behaviour.

### Added

- **Calibrated operating points.** A checkpoint may carry its own per-rank
  thresholds and a species gate (`operating_point` entry). They are used when
  no threshold is given, on the command line and in `taxonova.classify()`.
  The species cut-offs are the F1-optimal values fitted on validation data,
  and the genus cut-offs are fitted the same way; domain to family stay at 0.5. The values of each model
  are listed in [docs/models.md](docs/models.md#operating-points). The run log
  states which thresholds were used and where they came from.
- **Species abstention gate.** The four Greengenes2 models (`v4`, `v3v4`,
  `v5v7`, `16s`) withhold the species call
  when the read's species feature is far from the training records of the
  predicted species (Mahalanobis distance to the species centroid). The gate
  can only remove a species call; it never changes another rank. It applies
  whenever the model carries one, also when you set thresholds yourself.
- **V4-V5 models** for Greengenes2 (`v4v5`), SILVA 138.2
  (`v4v5-silva`) and RDP 19 (`v4v5-rdp`), for amplicons made with the primers
  515F-Y and 926R, with status `stable`: they are listed by `taxonova models`
  and included in `taxonova models download all`. They have no species gate.
  In these models a species group label (`[a|b|...]`, two or more species that
  the region cannot separate) is reported at the genus: the species column
  reads `Unclassified`. See the known limitations below.
- `--strand both` (and `strand="both"` in `taxonova.classify()`) classifies
  each read in the orientation the model is more confident about. The default
  stays `plus`; Taxonova now warns when many of the first reads score higher
  reverse-complemented. With `--strand both`, the species gate is computed in
  the orientation each read was classified in.
- `--label-rules`: optional species-rank rules. A species group may not span
  several genera, the species must belong to the genus call (the genus is
  never rewritten from the species), and an open-nomenclature placeholder is
  not reported as a species when a named alternative exists.
  `--genus-token-check auto|on|off` controls whether the species name must
  carry the genus name; `auto` uses the model's setting. The four
  Greengenes2 models for V4, V3-V4, V5-V7 and the whole gene carry a table of renamed genera (for example a species kept under its
  earlier genus name) that the check accepts.
- FASTQ input (plain or gzip-compressed). gzip is recognised from the file
  content, not only from a `.gz` name.
- `taxonova.labels`: label-hygiene functions (group labels, rank-path checks,
  placeholder names, ambiguity runs in reference records) and helpers that
  combine the network genus with nearest-neighbour hits, for pipelines built
  on Taxonova. `taxonova classify` does not run a nearest-neighbour step.

### Changed

- `--confidence` has no fixed default any more: without it (and without any
  `--<rank>-threshold`) the model's operating point is used, and a model
  without one keeps the 0.5 / 0.7 / 0.8 profile. Giving `--confidence` or any
  per-rank threshold restores the scaled profile of earlier releases.
- **`taxonova.classify()` default threshold (behaviour change).** The
  `confidence` argument now defaults to `None` instead of `0.8`, in
  `taxonova.classify()` and `taxonova.api.classify()`. With no `confidence`
  and no per-rank threshold, the model's calibrated operating point is used,
  exactly as by `taxonova classify`; a model without one (the 26.10 files,
  the operon model, your own models) still uses 0.8 with the scaled profile.
  An explicit `confidence=` or per-rank threshold replaces the operating
  point, as before. Migration: pass `confidence=0.8` to keep the earlier
  Python thresholds (the species gate of the four Greengenes2 models still
  applies); to reproduce 26.10 tables, also use the 26.10 files
  (`model="v4-26.10"`, ...).
- `taxonova classify ... --confidence 0` no longer reports a species for every
  read with the four Greengenes2 models (`v4`, `v3v4`, `v5v7`, `16s`),
  because the species gate still applies.
- Reads with too little sequence information (fewer than 100 valid bases, or
  more than 5% ambiguous or other characters) get no call at any rank. Before,
  an all-N read could receive a full lineage.
- Abstention always runs from the top: once a rank is below its threshold,
  every rank below it is `Unclassified`, whatever the per-rank thresholds.
- Checkpoints are loaded with the network width they were saved with, so
  models with a hidden size other than 512 load correctly.
- The old model keys (`gg2_v4`, `gg2_v3v4`, `gg2_v5v7`, `gg2_v1v9`) name the
  current Greengenes2 models. A file cached by 26.10 under its 26.10 name is
  used only when that 26.10 model is asked for.

### Fixed

- A FASTA record with an empty sequence was dropped, so the output had fewer
  rows than the input. Every record now gets a row.
- An empty file, or a file that is neither FASTA nor FASTQ, wrote a header-only
  table and exited with status 0; it is now an error with a one-line message.
- A read with no valid k-mer (all N, empty, or shorter than k) received a
  confident lineage; it is now `Unclassified` with confidence 0.
- Alignment gaps (`-`, `.`) and whitespace inside a sequence broke k-mers; they
  are removed, and the sequence is read in upper case. A non-ASCII symbol is
  read as N instead of being deleted.
- A rank above its threshold could follow an `Unclassified` rank when a parent
  threshold was set higher than the species threshold.
- When the species head rewrote a parent rank, the parent kept the confidence
  of the label the network had predicted there; it is now capped at the
  confidence of the rank that drove the rewrite.
- A header that is not valid UTF-8 no longer stops the run, a `>` line with no
  id is an error that names the line, and duplicate ids are kept (with a
  warning).

### Known limitations

- **V4-V5 models, species.** Reporting group labels at the genus was decided
  after the benchmark results were read, and was then tested once on a
  pre-specified set of in-silico V4-V5 amplicons from genomes new in GTDB
  R226. The test did not meet its pre-specified criterion, a species F1 gain
  whose 95% CI lies above zero in all three references: Greengenes2 +0.048
  (95% CI 0.027 to 0.069) and RDP +0.059 (0.046 to 0.072) met it, SILVA +0.006
  (-0.043 to 0.035) did not. Genus F1 was unchanged in all three. The test scored the
  evaluation pipeline, which adds a nearest-neighbour species step to the
  network; `taxonova classify` uses the network alone.
- **V4-V5 models, species gate.** None of the three V4-V5 models carries a
  species abstention gate (no Mahalanobis distance check), unlike the four
  Greengenes2 models for V4, V3-V4, V5-V7 and the whole gene, so a V4-V5
  species call is not withheld for reads far from the predicted species.
- **V4-V5 Greengenes2 model, label rules.** It carries no genus synonym table
  and has the genus-token check off, unlike the other Greengenes2 models.
  This only matters with `--label-rules`.
- **SILVA species.** This release has no SILVA model for V4, V3-V4, V5-V7 or
  the whole gene. In the benchmark for this release, the pre-registered SILVA
  configuration (network, nearest-neighbour search and a stacked genus step;
  not part of the package) gained genus F1 but lost species F1 against the
  same configuration refitted without the new genus step: V4 -0.051 (95% CI
  -0.070 to -0.032), V3-V4 -0.020 and V5-V7 -0.021 at species, +0.017 to
  +0.028 at genus. Species assignment for SILVA will be revisited together
  with the group-label convention.
- In the held-out benchmark for this release, some evaluation sequences also
  occurred in the calibration set; excluding them did not change any conclusion.
- One-member bracketed names such as `[Eubacterium]` are treated as group
  labels by `taxonova.labels` (recorded as an expected failure in the tests).

### Compatibility

- The 26.11 checkpoints load with `torch.load(..., weights_only=True)`, the
  default from torch 2.6, as the 26.10 checkpoints do. Under torch 2.5 and
  earlier, `weights_only=True` refuses them (and the 26.10 files) because the
  taxonomy table holds Python sets; Taxonova itself loads them on every
  supported torch version.
- Checkpoints without an operating point (your own models, the 26.10 files,
  the operon model) behave as before.
- New downloads: about 250 to 270 MB per Greengenes2 model and 207 to 524 MB
  per V4-V5 model, once.

## [26.10.3] - 2026-10-30

Withdraws 26.10.2. Code, models and output are the same as 26.10.1.

### Removed

- The four preview models added in 26.10.2 (`v4-rdp`, `v3v4-rdp`, `v5v7-rdp`,
  `v3v4-silva`). Their species-level performance needs more evaluation before
  they are offered; SILVA and RDP models remain in preparation. Use 26.10.3,
  not 26.10.2.

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
  continuous integration on Python 3.10 to 3.12 and a release workflow.

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
- Python 3.10 or later is required. Python 3.8 and 3.9 have reached end of life
  and are no longer tested; taxonova 26.10.0 still installs on them.
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
