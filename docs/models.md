# Models

Pick the model that matches how your sequences were made. Each one is
downloaded the first time you use it (200 to 520 MB) and kept for later runs.

| id | amplicon | reference | taxonomy | model version | status |
| --- | --- | --- | --- | --- | --- |
| `v4` | 16S V4 (about 250 bp) | Greengenes2 2024.09 | GTDB, through Greengenes2 | 26.11 | stable |
| `v3v4` | 16S V3-V4 (about 450 bp) | Greengenes2 2024.09 | GTDB, through Greengenes2 | 26.11 | stable |
| `v5v7` | 16S V5-V7 | Greengenes2 2024.09 | GTDB, through Greengenes2 | 26.11 | stable |
| `16s` | whole 16S gene, V1-V9 | Greengenes2 2024.09 | GTDB, through Greengenes2 | 26.11 | stable |
| `operon` | rRNA operon, 16S-ITS-23S | GROND (GTDB R207 build) | GTDB R207 | 1.0 | stable |
| `v4v5` | 16S V4-V5 (515F-Y / 926R) | Greengenes2 2024.09 | GTDB, through Greengenes2 | 26.11 | stable |
| `v4v5-silva` | 16S V4-V5 (515F-Y / 926R) | SILVA 138.2 | SILVA 138.2 | 26.11 | stable |
| `v4v5-rdp` | 16S V4-V5 (515F-Y / 926R) | RDP 19 | RDP 19 | 26.11 | stable |

Read the
[known limitations](https://github.com/kardokhk/taxonova/blob/main/CHANGELOG.md#known-limitations)
of the V4-V5 models before you use them: they have no species gate, and the
V4-V5 Greengenes2 model has the genus-token check off. SILVA and RDP models
for the other regions are in preparation.

```bash
taxonova models                      # list the models and which are downloaded
taxonova models download v4 16s      # download in advance (for example on a login node)
taxonova models --all                # also list preview and deprecated models
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

- `https://github.com/kardokhk/taxonova/releases/download/<tag>/<file>`, where
  `<tag>` is `v26.11.0` for the 26.11 files and `v26.10.1` for the 26.10 files
  and the operon model
- `https://huggingface.co/claessonlab/taxonova-models/resolve/main/<file>`

You can also pass the path to any `.pt` file with `--model`.

## Files and checksums

| id | file | sha256 | md5 |
| --- | --- | --- | --- |
| `v4` | `taxonova-V4-greengenes2-26.11.pt` | `2026db6c5d4875247d7f581c24bd3ae45c43281a5fd1d13de9f95caa8c4078c2` | `e0058514d575d5d2c9d33728deeac8cc` |
| `v3v4` | `taxonova-V3V4-greengenes2-26.11.pt` | `13eadc5cbefbf511ed364244101951e3a5b10d2a85318ca621cc732ae75eee5b` | `d6d73e8f5d032c6ad2cc95a8731e9368` |
| `v5v7` | `taxonova-V5V7-greengenes2-26.11.pt` | `0156510cc97ca10ef30676257fcc5e190257d1f1a5cf9b11bdd35afdf558f354` | `c9288833baa412f3702e1299ae7cd26b` |
| `16s` | `taxonova-16S-greengenes2-26.11.pt` | `90cac67a43a95350c1587a72cd017249442ef6214d8bc32a7a7aaa08d8b36cb1` | `757a638b2ed022bf4743d6343c44c963` |
| `operon` | `taxonova-operon-grond-1.0.pt` | `9ac36fea212edd569198027c9e6fb182199bb316acacad188d758929c9408289` | `6f5d4ef039888a33ad765ef41725b624` |
| `v4v5` | `taxonova-V4V5-greengenes2-26.11.pt` | `68bfdeada873e325299547ade195025a39370f122674b13a5ae14fa273058ebc` | `d67b3b3115fc089f1993540dc5283bc2` |
| `v4v5-silva` | `taxonova-V4V5-silva-26.11.pt` | `c358924f3c9fc2911e6415949fa8771ab45ae67d1fb6df017fe002d3689cac4e` | `eb8ea861d31852ba9a24ce1140002855` |
| `v4v5-rdp` | `taxonova-V4V5-rdp-26.11.pt` | `1b62cbf91817803d6410882f3e231b7d9442957074e89bbc072c87b46a5aef8b` | `b786769f384abeeb25620f950ab6e34b` |
| `v4-26.10` | `taxonova-V4-greengenes2-26.10.pt` | `c52aaafe2b9b79092171c11f67de469ac9d64f3e8fadac74e63e756792c0a3a3` | `bccf86c9086b68defe8bc8a2a3dff2b6` |
| `v3v4-26.10` | `taxonova-V3V4-greengenes2-26.10.pt` | `005677c10327b981847b9ec5c5c529828716d0d628706fa65e2f77222309e0aa` | `a5e584504627a8055afe25fd56893e37` |
| `v5v7-26.10` | `taxonova-V5V7-greengenes2-26.10.pt` | `77f7a75bf38c4923422dadf88e971764a3ccf604b23c1c96b46c0b325a670385` | `6de1add678c05fd0c5c3679ea4094d1d` |
| `16s-26.10` | `taxonova-16S-greengenes2-26.10.pt` | `d2ec6d014cf771525887c136fa0591b09d5284db6035022bb60f7f96d2e453ca` | `103e508682ad0b948e9f8a993bb7be0a` |

The same values are in `taxonova/models.py`, which the downloader reads. The
26.11 Greengenes2 files hold the same network weights as the 26.10 files; only
the operating point, the species centroid table and the label-rule settings
were added.

## Operating points

A model's operating point is used when you give no threshold (no
`--confidence` and no `--<rank>-threshold`). Domain to family use 0.5 in every
model below. With any threshold on the command line, the scaled profile of
[output.md](output.md#thresholds) is used instead; the species gate still
applies. `taxonova.classify()` in Python uses the same rule.

| id | genus | species | species gate | species group labels | label rules (`--label-rules`) |
| --- | --- | --- | --- | --- | --- |
| `v4` | 0.56 | 0.50 | Mahalanobis, at most 1636.8 | kept | genus-token check on, renamed-genus table |
| `v3v4` | 0.50 | 0.48 | Mahalanobis, at most 2080.5 | kept | genus-token check on, renamed-genus table |
| `v5v7` | 0.61 | 0.51 | Mahalanobis, at most 1706.5 | kept | genus-token check on, renamed-genus table |
| `16s` | 0.53 | 0.47 | Mahalanobis, at most 3757.0 | kept | genus-token check on, renamed-genus table |
| `v4v5` | 0.52 | 0.50 | none | reported at the genus | genus-token check off |
| `v4v5-silva` | 0.45 | 0.69 | none | reported at the genus | genus-token check on |
| `v4v5-rdp` | 0.69 | 0.59 | none | reported at the genus | genus-token check on |

- Genus and species cut-offs maximise F1 on validation data of the same
  reference and region (the median over three training seeds of the recipe).
  For the V4-V5 models the species cut-off was refitted with group labels
  reported at the genus.
- The species gate withholds the species call when the squared Mahalanobis
  distance of the read's species feature to the training centroid of the
  predicted species is above the listed value. It only removes species calls.
- "Reported at the genus": a species group label (`[a|b|...]`) becomes
  `Unclassified` at species, and the genus call stands.

## How the models were made

The Greengenes2 V4, V3-V4, V5-V7 and whole-16S models and the operon model
use the architecture and training recipe described in the
[paper](https://doi.org/10.1371/journal.pcbi.1014912). These four 16S models are
trained on Greengenes2 release 2024.09, which takes its species names from GTDB.
The operon model is trained on GROND, built for GTDB release 207. The V4-V5
models are trained with the same architecture on in-silico V4-V5 amplicons of
each reference (primers 515F-Y and 926R); the Greengenes2 and SILVA V4-V5
networks are twice as wide (hidden size 1024), and the RDP network was started
from a SILVA V4-V5 network. The primers, extraction settings and training-set
checksums are stored in each V4-V5 checkpoint (`region_metadata`). The history
of the 26.10 checkpoints, and how output changed when they replaced those of
v1.0.0, is in the
[changelog](https://github.com/kardokhk/taxonova/blob/main/CHANGELOG.md#26100---2026-10-29).
You do not need to calibrate the pretrained models; `taxonova calibrate` and
`taxonova fit-ood` are for models you train yourself.

## Earlier names

Earlier ids and file names keep working, with no warning. An earlier file
name of the same bytes (the 26.10.0 asset names, the operon file of v1.0.0)
finds a copy you already downloaded, so it is not fetched again.

| id | also accepted |
| --- | --- |
| `v4` | `gg2_v4`, `V4` |
| `v3v4` | `gg2_v3v4`, `V3-V4` |
| `v5v7` | `gg2_v5v7`, `V5-V7` |
| `16s` | `gg2_v1v9`, `v1v9`, `V1-V9`, `full` |
| `operon` | `gtdb_rrn`, `rrn`, `RRN`, `gtdb_rrn_mrgroups_model.pt` |
| `v4v5` | `gg2_v4v5`, `V4-V5` |
| `v4-26.10` | `taxonova-V4-greengenes2-26.10.pt`, `gg2_v4_mrgroups_model_26.10.pt` |
| `v3v4-26.10` | `taxonova-V3V4-greengenes2-26.10.pt`, `gg2_v3v4_mrgroups_model_26.10.pt` |
| `v5v7-26.10` | `taxonova-V5V7-greengenes2-26.10.pt`, `gg2_v5v7_mrgroups_model_26.10.pt` |
| `16s-26.10` | `taxonova-16S-greengenes2-26.10.pt`, `gg2_v1v9_mrgroups_model_26.10.pt` |

Since 26.11.0 the short ids and the old keys name the 26.11 files. The 26.10
file names resolve to the 26.10 checkpoints (status `deprecated`): a run
that names one of them uses the 0.5 / 0.7 / 0.8 default and no species gate.

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
   `status="preview"`, the file name, `urls=_urls(filename, tag="vYY.MM.MICRO")`
   with the tag of the release that will carry the file, the checksums,
   the size in MB, the reference and its version, and a one-line description.
4. Upload the file to the GitHub release named by that tag (the release of
   the package version that adds the model) and to the Hugging Face mirror `claessonlab/taxonova-models` (see
   [releasing.md](releasing.md)), and add it to the mirror's model card.
5. Add a line to this page and to the changelog. When the model has been
   evaluated, change its status to `stable` in a later release; it then appears
   in `taxonova models`, in the README table and in the web app.
