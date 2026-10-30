# Models

Pick the model that matches how your sequences were made and the reference
whose names you want. Each model is downloaded the first time you use it
(about 210 to 250 MB) and kept for later runs.

| id | amplicon | reference | taxonomy | model version |
| --- | --- | --- | --- | --- |
| `v4` | 16S V4 (about 250 bp) | Greengenes2 2024.09 | GTDB, through Greengenes2 | 26.10 |
| `v3v4` | 16S V3-V4 (about 450 bp) | Greengenes2 2024.09 | GTDB, through Greengenes2 | 26.10 |
| `v5v7` | 16S V5-V7 | Greengenes2 2024.09 | GTDB, through Greengenes2 | 26.10 |
| `16s` | whole 16S gene, V1-V9 | Greengenes2 2024.09 | GTDB, through Greengenes2 | 26.10 |
| `operon` | rRNA operon, 16S-ITS-23S | GROND (GTDB R207 build) | GTDB R207 | 1.0 |

Preview models (see [Preview models](#preview-models) before you use them):

| id | amplicon | reference | taxonomy | model version |
| --- | --- | --- | --- | --- |
| `v4-rdp` | 16S V4 (515F/806R) | RDP training set 19 | RDP | 26.10 |
| `v3v4-silva` | 16S V3-V4 (341F/805R) | SILVA 138.2 SSU Ref NR99 | SILVA | 26.10 |
| `v3v4-rdp` | 16S V3-V4 (341F/805R) | RDP training set 19 | RDP | 26.10 |
| `v5v7-rdp` | 16S V5-V7 (799F/1193R) | RDP training set 19 | RDP | 26.10 |

A model id is `<region>-<reference>`; without a suffix it is the Greengenes2
model of that region. The output of a model uses the names of its reference,
so tables made with different references cannot be compared name by name.

```bash
taxonova models                      # list the models and which are downloaded
taxonova models download v4 16s      # download in advance (for example on a login node)
taxonova models download all         # the stable models; name preview models by id
taxonova classify reads.fasta --model v3v4
taxonova classify reads.fasta --model v3v4-silva
```

## Reference databases

| reference | version | used by | status | names | source |
| --- | --- | --- | --- | --- | --- |
| Greengenes2 | 2024.09 | `v4`, `v3v4`, `v5v7`, `16s` | stable, the default | GTDB, through Greengenes2 | Greengenes2 2024.09 backbone |
| SILVA | 138.2, SSU Ref NR99 | `v3v4-silva` | preview | SILVA | DADA2 training set by B. Callahan, [doi:10.5281/zenodo.14169026](https://doi.org/10.5281/zenodo.14169026), CC BY 4.0 |
| RDP | training set 19 (2023-08-23) | `v4-rdp`, `v3v4-rdp`, `v5v7-rdp` | preview | RDP | DADA2 training set by B. Callahan, [doi:10.5281/zenodo.14168771](https://doi.org/10.5281/zenodo.14168771), CC BY 4.0 |
| GROND | GTDB R207 build | `operon` | stable | GTDB R207 | as in the paper |

If you use a SILVA model, please also cite Quast C, et al. The SILVA ribosomal
RNA gene database project: improved data processing and web-based tools.
Nucleic Acids Res. 2013;41(D1):D590-D596.
[doi:10.1093/nar/gks1219](https://doi.org/10.1093/nar/gks1219).
If you use an RDP model, please also cite Wang Q, et al. Naive Bayesian
classifier for rapid assignment of rRNA sequences into the new bacterial
taxonomy. Appl Environ Microbiol. 2007;73(16):5261-5267.
[doi:10.1128/AEM.00062-07](https://doi.org/10.1128/AEM.00062-07), and Cole JR,
et al. Ribosomal Database Project: data and tools for high throughput rRNA
analysis. Nucleic Acids Res. 2014;42(D1):D633-D642.
[doi:10.1093/nar/gkt1244](https://doi.org/10.1093/nar/gkt1244).

SILVA, RDP and Greengenes2 differ in names and in how they split some genera.
For example, the broad *Prevotella* of Greengenes2 2024.09 is divided in
SILVA 138.2 and RDP 19 into *Segatella*, *Hoylesella*, *Xylanibacter* and
others. Choose the reference whose names you will report, and do not mix
references within one analysis.

Not released yet: SILVA V4 and V5-V7 (genus precision on held-out sequences
below the release bar), and whole-16S models for SILVA and RDP (the mock
evaluation used here cannot assess whole-gene reads). They may follow in a
later release.

## Where models are stored

In `~/.taxonova/models/`, or in the directory named by the
`TAXONOVA_MODELS_DIR` environment variable. On a cluster, point it at shared
project storage so that every job uses the same copy:

```bash
export TAXONOVA_MODELS_DIR=/path/to/shared/taxonova-models
```

A file in that directory is used only if its sha256 matches the published
value; otherwise it is reported and downloaded again. Downloads are written to
`<file>.part` and renamed only after the checksums pass.

Each model is downloaded from the GitHub release first and, if that fails,
from the mirror on Hugging Face. To install by hand (for example on a machine
without internet access), download the file from either address, put it in the
models directory, and check it with `sha256sum`:

- `https://github.com/kardokhk/taxonova/releases/download/<tag>/<file>`,
  where `<tag>` is `v26.10.1` for the stable models and `v26.10.2` for the
  preview models
- `https://huggingface.co/claessonlab/taxonova-models/resolve/main/<file>`

You can also pass the path to any `.pt` file with `--model`.

## Files and checksums

| id | file | sha256 | md5 |
| --- | --- | --- | --- |
| `v4` | `taxonova-V4-greengenes2-26.10.pt` | `c52aaafe2b9b79092171c11f67de469ac9d64f3e8fadac74e63e756792c0a3a3` | `bccf86c9086b68defe8bc8a2a3dff2b6` |
| `v3v4` | `taxonova-V3V4-greengenes2-26.10.pt` | `005677c10327b981847b9ec5c5c529828716d0d628706fa65e2f77222309e0aa` | `a5e584504627a8055afe25fd56893e37` |
| `v5v7` | `taxonova-V5V7-greengenes2-26.10.pt` | `77f7a75bf38c4923422dadf88e971764a3ccf604b23c1c96b46c0b325a670385` | `6de1add678c05fd0c5c3679ea4094d1d` |
| `16s` | `taxonova-16S-greengenes2-26.10.pt` | `d2ec6d014cf771525887c136fa0591b09d5284db6035022bb60f7f96d2e453ca` | `103e508682ad0b948e9f8a993bb7be0a` |
| `operon` | `taxonova-operon-grond-1.0.pt` | `9ac36fea212edd569198027c9e6fb182199bb316acacad188d758929c9408289` | `6f5d4ef039888a33ad765ef41725b624` |
| `v4-rdp` | `taxonova-V4-rdp-26.10.pt` | `785a94a6dc710a4feb277ac291ff7d1640ed9e4d86feaf7ace61b3d152135efd` | `f3974e9a0c0b0aa0a4c2a03af4a3a80a` |
| `v3v4-silva` | `taxonova-V3V4-silva-26.10.pt` | `929a56bf14ed2937e5571b933459cab07d3231a73c76901367406c595b59f94d` | `93f65881154b927a99277d25e6be6e15` |
| `v3v4-rdp` | `taxonova-V3V4-rdp-26.10.pt` | `199e653704a47a32b5b49ba7d4db5a6129d41ffcf75bcb39abed29b0f5d1e36e` | `53de1b25c6837b560361cb879d3ab59d` |
| `v5v7-rdp` | `taxonova-V5V7-rdp-26.10.pt` | `b0b936cc93726bec605d41f375321d0dfaaa27a7db03378d275ef10ffb52d8f8` | `6e72ceadec08776bce539f167d53b755` |

The same values are in `taxonova/models.py`, which the downloader reads.

## How the models were made

The five stable models use the architecture and training recipe described in
the [paper](https://doi.org/10.1371/journal.pcbi.1014912). The preview models
are described in the next section. The four 16S models are
trained on Greengenes2 release 2024.09, which takes its species names from GTDB.
The operon model is trained on GROND, built for GTDB release 207. The history
of the 26.10 checkpoints, and how output changed when they replaced those of
v1.0.0, is in the
[changelog](https://github.com/kardokhk/taxonova/blob/main/CHANGELOG.md#26100---2026-10-29).
You do not need to calibrate the pretrained models; `taxonova calibrate` and
`taxonova fit-ood` are for models you train yourself.

## Preview models

The four preview models were trained on 2026-10-30 and evaluated on held-out
reference sequences and on the TaxoBench mock communities. They are released
as previews because the evaluation is not complete: they have not been tested
on real-world samples or on taxa absent from their reference, one training
seed was used per model, and their confidence thresholds are the package
defaults, not fitted to them. The rules for every number below were written
down before the corresponding models were run, with the exception stated under
[Mock communities](#mock-communities).

### How they were trained

- **Data.** The DADA2-formatted training sets of SILVA 138.2 SSU Ref NR99 and
  RDP training set 19 (see [Reference databases](#reference-databases)).
  Eukaryota were removed; placeholder names that would merge unrelated clades
  (SILVA `Incertae Sedis`, `Unknown Family`) were qualified with the nearest
  named ancestor; species are `Genus epithet`, and non-binomial epithets
  (`sp.`, `bacterium` and similar) were cleared.
- **Regions.** Amplicons were cut from the full-length records by in-silico PCR
  with the primers in the table above (cutadapt, error rate 0.2 per primer,
  both strands), keeping V4 at 200 to 350 bp, V3-V4 at 400 to 500 bp and
  V5-V7 at 300 to 500 bp. Identical sequences were merged; where they carry
  different names, the model learns a grouped label `[a|b|...]`, as for the
  Greengenes2 models.
- **Held-out split.** Before training, up to 5% of records of each reference
  (SILVA 2.6%, RDP 5.0%) were set aside, at most one record per lineage with
  at least two records, so every held-out record has a training record with
  the same full lineage. These records were not used in training.
- **Training.** The same network, settings and code as the Greengenes2 models
  (hierarchical architecture, hidden size 512, batch 256, learning rate 0.001,
  8-mers), seed 1001 fixed in advance. Epochs as for Greengenes2 (V4 45,
  V3-V4 20, V5-V7 90). Because the RDP sets are 6 to 17 times smaller, a
  second RDP run per region was trained with the number of optimiser steps
  matched to Greengenes2, and a rule fixed in advance chose between the two on
  held-out data: the first run for V4 and V5-V7, the step-matched run for V3-V4
  (stopped early at epoch 160).

| id | training sequences (after merging) | genus / species classes |
| --- | --- | --- |
| `v4-rdp` | 14,867 | 3,868 / 13,742 |
| `v3v4-silva` | 272,210 | 4,432 / 12,903 |
| `v3v4-rdp` | 16,644 | 3,477 / 14,910 |
| `v5v7-rdp` | 17,299 | 3,545 / 15,555 |

Class counts include grouped labels and the empty label.

### Held-out sequences

Held-out sequences that are not identical to any training sequence, classified
with `taxonova classify` at the default thresholds. A call counts as correct
when it equals the true name or is a grouped label that contains it. A row is
counted as a misclassification at a rank when its deepest call is wrong at
that rank or above. Precision and recall with 95% confidence intervals
(percentile bootstrap, 2,000 resamples).

| id | n (genus / species) | genus precision | genus recall | species precision | species recall |
| --- | --- | --- | --- | --- | --- |
| `v4-rdp` | 345 / 339 | 0.984 (0.969 to 0.997) | 0.901 (0.870 to 0.930) | 0.918 (0.876 to 0.956) | 0.463 (0.410 to 0.516) |
| `v3v4-silva` | 4,493 / 2,049 | 0.936 (0.928 to 0.943) | 0.849 (0.839 to 0.859) | 0.852 (0.827 to 0.877) | 0.332 (0.312 to 0.352) |
| `v3v4-rdp` | 457 / 450 | 0.987 (0.976 to 0.996) | 0.978 (0.963 to 0.991) | 0.870 (0.837 to 0.900) | 0.758 (0.716 to 0.796) |
| `v5v7-rdp` | 505 / 500 | 0.978 (0.963 to 0.990) | 0.952 (0.933 to 0.970) | 0.864 (0.830 to 0.897) | 0.662 (0.620 to 0.702) |

Species calls on sequences unlike those in the reference are less reliable
than genus calls: species precision is 0.85 to 0.92 and species recall 0.33 to
0.76 here, and no model met the pre-set species bar (lower confidence bound of
precision at least 0.90 and of recall at least 0.50). The RDP sets are small
(339 to 505 sequences), hence the wider intervals. SILVA and RDP numbers are
not a ranking of the references: the held-out sets and label sets differ. No
Greengenes2 model has a comparable held-out set.

### Mock communities

The 13 TaxoBench mock communities at each region. Primary analysis (native
truth): the expected name of each mock sequence was taken from the model's own
reference (the TaxoBench truth rule: best match within 3 mismatches, common
ancestor of ties), and only sequences with such a match were scored. Medians
across the 13 communities at the default thresholds; worst community among
those with at least 20 scored sequences. Greengenes2 is the stable model of the
same region, scored the same way.

| id | sequences with a reference match, median | genus F1, median (range) | worst-community genus precision | species F1, median | worst-community species precision | Greengenes2 genus F1, median |
| --- | --- | --- | --- | --- | --- | --- |
| `v4-rdp` | 0.830 | 0.984 (0.965 to 1.000) | 0.989 | 0.900 | 0.980 | 0.980 |
| `v3v4-silva` | 0.875 | 0.982 (0.959 to 1.000) | 0.966 | 0.847 | 0.917 | 0.972 |
| `v3v4-rdp` | 0.679 | 1.000 (0.981 to 1.000) | 0.981 | 0.930 | 0.921 | 0.972 |
| `v5v7-rdp` | 0.735 | 1.000 (0.950 to 1.000) | 0.960 | 0.971 | 0.903 | 1.000 |

Read these numbers with two limits in mind:

- **Most scored mock sequences are in the reference.** Of the mock sequences
  with a reference match, 71% to 85% are identical to a training sequence.
  The mock results therefore show agreement with the reference's own names for
  well-described organisms; the held-out results above are the test on unseen
  sequences.
- **Sequences without a reference match** (a median 12% to 32% of mock
  sequences for these models) still receive a genus name in a median 50% to
  74% of cases for the RDP models and 83% for `v3v4-silva`. Whether those names
  are right cannot be judged from this truth.

**Order of the analyses.** The mock evaluation was first registered with a
name-matching rule (rule A): SILVA and RDP calls were scored against the
TaxoBench truth, which uses Greengenes2 (GTDB) names, after removing GTDB
suffixes such as `_A`. Those results were committed at 15:16 on 2026-10-30.
Under rule A no model met the release criterion. At 15:18 the evaluation plan
was amended to make native truth the primary analysis. The change was the
author's decision that SILVA and RDP names should not be made equal to GTDB
names, because a user who chooses a reference reads that reference's names. It
was made after the rule A results were known. Rule A is kept as a secondary
analysis:

| id | genus F1, median (range), rule A | worst-community genus precision, rule A | Greengenes2 genus F1, median, rule A |
| --- | --- | --- | --- |
| `v4-rdp` | 0.908 (0.356 to 1.000) | 0.381 | 0.993 |
| `v3v4-silva` | 0.895 (0.447 to 1.000) | 0.460 | 0.971 |
| `v3v4-rdp` | 0.924 (0.486 to 1.000) | 0.517 | 0.971 |
| `v5v7-rdp` | 0.925 (0.500 to 1.000) | 0.524 | 1.000 |

The worst community under rule A is the gut mock in every region: its truth
genus is *Prevotella* in the broad Greengenes2 sense, which SILVA and RDP
split (for example truth *Prevotella copri*, call *Segatella copri*). Rule A
cannot resolve a split genus, so these calls count as errors. How much of the
remaining gap to Greengenes2 under rule A is naming and how much is model error
cannot be separated with rule A.

## Earlier names

Earlier ids and file names keep working, with no warning. The files are the
same bytes under a new name, so a copy you already downloaded is used and not
fetched again.

| id | also accepted |
| --- | --- |
| `v4` | `gg2_v4`, `V4`, `gg2_v4_mrgroups_model_26.10.pt` |
| `v3v4` | `gg2_v3v4`, `V3-V4`, `gg2_v3v4_mrgroups_model_26.10.pt` |
| `v5v7` | `gg2_v5v7`, `V5-V7`, `gg2_v5v7_mrgroups_model_26.10.pt` |
| `16s` | `gg2_v1v9`, `v1v9`, `V1-V9`, `full`, `gg2_v1v9_mrgroups_model_26.10.pt` |
| `operon` | `gtdb_rrn`, `rrn`, `RRN`, `gtdb_rrn_mrgroups_model.pt` |

Names are matched without regard to case. The v26.10.0 and v1.0.0 release
assets stay where they are, so scripts that download them by URL keep working.

The Greengenes2 checkpoints of v1.0.0 (`gg2_v4_mrgroups_model.pt`,
`gg2_v3v4_mrgroups_model.pt`, `gg2_v5v7_mrgroups_model.pt`,
`gg2_v1v9_mrgroups_model.pt`) are superseded and are not aliases. If you pass
the path to one of them, Taxonova loads it and prints a warning that names its
replacement. They remain on the
[v1.0.0 release](https://github.com/kardokhk/taxonova/releases/tag/v1.0.0)
so that earlier analyses can be traced.

## Versioning

- A model id names what a model is for: `<region>[-<reference>]`, for example
  `v4` or `v3v4-silva`. The reference is left out for
  the default reference of a region (Greengenes2 for the 16S regions).
- A file name is `taxonova-<REGION>-<reference>-<model version>.pt`. The model
  version is the year and month the checkpoint was made (`26.10`); it is not
  the package version. A file never changes once released: a new checkpoint
  gets a new file name.
- An id points to the recommended checkpoint for that region and reference.
  When a better checkpoint replaces it, the changelog says so, says how output
  changes, and the earlier file stays downloadable for reproducibility.
- Status: `stable` models have been evaluated and are the default choice.
  `preview` models are released and can be used by id; `taxonova models`
  lists them in a separate section, and they print a note when used. A preview
  model may be replaced by a new checkpoint (under a new file name) before it
  becomes stable. `deprecated` models are kept so that earlier analyses can be
  rerun and are listed only by `taxonova models --all`.
- For reproducible work, record the output of `taxonova --version` and the
  model file name. Both are in the run log written next to the output.

## Adding a model

For maintainers. A new model (for example `v4-silva`) is added in one place,
`taxonova/models.py`, and released with the package.

1. Train and check the checkpoint (see [training.md](training.md)). Record the
   reference release, the command, the seed and the package commit.
2. Name the file `taxonova-<REGION>-<reference>-<YY.MM>.pt` and compute its
   checksums: `sha256sum`, `md5sum`.
3. Add a `ModelSpec` to `_REGISTRY` in `taxonova/models.py` with
   `status="preview"`, the file name, `urls=_urls(filename, tag="vYY.MM.MICRO")`
   with the tag of the release that will carry the file, the checksums,
   the size in MB, the reference and its version, and a one-line description.
4. Upload the file to the GitHub release named by that tag (the release of
   the package version that adds the model) and to the Hugging Face mirror `claessonlab/taxonova-models` (see
   [releasing.md](releasing.md)), and add it to the mirror's model card.
5. Add a line to this page and to the changelog. When the model has been
   evaluated, change its status to `stable` in a later release; it then moves
   to the stable section of `taxonova models` and to the README table.
