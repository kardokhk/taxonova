# Models

Pick the model that matches how your sequences were made. Each one is
downloaded the first time you use it (about 240 MB) and kept for later runs.

| id | amplicon | reference | taxonomy | model version |
| --- | --- | --- | --- | --- |
| `v4` | 16S V4 (about 250 bp) | Greengenes2 2024.09 | GTDB, through Greengenes2 | 26.10 |
| `v3v4` | 16S V3-V4 (about 450 bp) | Greengenes2 2024.09 | GTDB, through Greengenes2 | 26.10 |
| `v5v7` | 16S V5-V7 | Greengenes2 2024.09 | GTDB, through Greengenes2 | 26.10 |
| `16s` | whole 16S gene, V1-V9 | Greengenes2 2024.09 | GTDB, through Greengenes2 | 26.10 |
| `operon` | rRNA operon, 16S-ITS-23S | GROND (GTDB R207 build) | GTDB R207 | 1.0 |

Models for SILVA and RDP are in preparation. They will appear here, and in
`taxonova models`, when they are released.

```bash
taxonova models                      # list the models and which are downloaded
taxonova models download v4 16s      # download in advance (for example on a login node)
taxonova models download all
taxonova classify reads.fasta --model v3v4
```

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

- `https://github.com/kardokhk/taxonova/releases/download/v26.10.1/<file>`
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

The same values are in `taxonova/models.py`, which the downloader reads.

## How the models were made

All five models use the architecture and training recipe described in the
[paper](https://doi.org/10.1371/journal.pcbi.1014912). The four 16S models are
trained on Greengenes2 release 2024.09, which takes its species names from GTDB.
The operon model is trained on GROND, built for GTDB release 207.

Model version 26.10 of the four Greengenes2 models replaced the v1.0.0
checkpoints in taxonova 26.10.0:

| model | how the 26.10 checkpoint was made |
| --- | --- |
| `16s`, `v5v7` | the v1.0.0 networks with tau-normalisation (tau 0.5) undone exactly |
| `v4`, `v3v4` | retrained on the same reference with the same recipe, without tau-normalisation or an out-of-distribution gate (seeds 1005 and 1006) |

None of the five models carries temperature scaling or an out-of-distribution
(`ood_config`) gate. Their confidences are the network's softmax outputs,
capped at the confidence of the rank above, and the thresholds you ask for are
applied to every sequence. You do not need to run `taxonova calibrate` or
`taxonova fit-ood` on them; those commands are for models you train yourself.
The reasons for the 26.10 replacement, and how output changed, are in the
[changelog](https://github.com/kardokhk/taxonova/blob/main/CHANGELOG.md#26100---2026-10-29).

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
  `v4` or, for a future SILVA model, `v4-silva`. The reference is left out for
  the default reference of a region (Greengenes2 for the 16S regions).
- A file name is `taxonova-<REGION>-<reference>-<model version>.pt`. The model
  version is the year and month the checkpoint was made (`26.10`); it is not
  the package version. A file never changes once released: a new checkpoint
  gets a new file name.
- An id points to the recommended checkpoint for that region and reference.
  When a better checkpoint replaces it, the changelog says so, says how output
  changes, and the earlier file stays downloadable for reproducibility.
- Status: `stable` models are listed by default. `preview` models can be used
  by id but are listed only by `taxonova models --all`, and print a note when
  used. `deprecated` models are kept so that earlier analyses can be rerun.
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
   `status="preview"`, the file name, `urls=_urls(filename)`, the checksums,
   the size in MB, the reference and its version, and a one-line description.
4. Upload the file to the GitHub release of the package version that adds it
   and to the Hugging Face mirror `claessonlab/taxonova-models` (see
   [releasing.md](releasing.md)), and add it to the mirror's model card.
5. Add a line to this page and to the changelog. When the model has been
   evaluated, change its status to `stable` in a later release; it then appears
   in `taxonova models`, in the README table and in the web app.
