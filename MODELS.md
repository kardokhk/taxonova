# Pretrained models

Taxonova ships five pretrained models, one for each amplicon region or operon.
Pick the one that matches how your sequences were generated, download it, and pass
it to `--model_input`. That is the whole workflow.

| Model file | Use it for | Reference taxonomy | Release asset |
| --- | --- | --- | --- |
| `gg2_v4_mrgroups_model_26.10.pt` | V4 amplicons | Greengenes2 | v26.10.0 |
| `gg2_v3v4_mrgroups_model_26.10.pt` | V3-V4 amplicons | Greengenes2 | v26.10.0 |
| `gg2_v5v7_mrgroups_model_26.10.pt` | V5-V7 amplicons | Greengenes2 | v26.10.0 |
| `gg2_v1v9_mrgroups_model_26.10.pt` | whole 16S (V1-V9) | Greengenes2 | v26.10.0 |
| `gtdb_rrn_mrgroups_model.pt` | RRN, the full rRNA operon | GROND, GTDB taxonomy | v1.0.0 |

The four Greengenes2 models take their species nomenclature from GTDB by way of
Greengenes2; the RRN model is trained on GROND, which uses GTDB taxonomy.

## Where they live

Each checkpoint is roughly 230 to 245 MB, too large to keep inside the git repo, so
they are attached to GitHub releases rather than committed. The four Greengenes2
checkpoints are attached to release v26.10.0 and the RRN checkpoint to v1.0.0. The
downloader checks each file against the md5 and sha256 listed in
`taxonova/utils/model_downloader.py`, and the v26.10.0 release carries a
`SHA256SUMS` file.

The Greengenes2 checkpoints of v1.0.0 (`gg2_<region>_mrgroups_model.pt`, without
the `_26.10` suffix) are superseded; see "Changes in 26.10.0" below. They remain on
the v1.0.0 release page so that earlier analyses can be traced, and Taxonova prints
a warning when one of them is loaded.

## Downloading and using one

The easy way is the built-in downloader, which fetches the checkpoints from the
GitHub Release for you. The model keys are `gg2_v4`, `gg2_v3v4`, `gg2_v5v7`,
`gg2_v1v9`, and `gtdb_rrn`.

```bash
# interactive picker (also runs as `taxonova-setup` straight after install)
taxonova-setup

# or pull specific models / everything
taxonova download-models --models gg2_v4
taxonova download-models --models all

# list what is available and what you already have
taxonova download-models --list
```

Models are stored in `~/.taxonova/models/` by default. A file already there is
used only if its checksums match; otherwise it is reported and downloaded again. Set the
`TAXONOVA_MODELS_DIR` environment variable to keep them somewhere else. Once a
model is in that directory you can pass just its filename to `--model_input` and
Taxonova will resolve it automatically:

```bash
taxonova classification \
    --input reads.fasta \
    --model_input gg2_v4_mrgroups_model_26.10.pt \
    --output predictions.tsv
```

You can still download the assets manually from the Releases page and pass a full
path to `--model_input` instead:

https://github.com/kardokhk/taxonova/releases

## Confidence calibration and the OOD gate

The 26.10.0 Greengenes2 checkpoints and the RRN checkpoint carry no
tau-normalisation, temperature scaling or out-of-distribution (`ood_config`)
gate. Their confidences are the model's softmax outputs, capped at the
confidence of the rank above, and the thresholds you request are applied to
every read. You do not need to run `taxonova calibrate` or `taxonova fit-ood` on
them; those modes are for models you train yourself.

## Changes in 26.10.0

Under taxonova 1.0.0 to 26.9.2 the v1.0.0 Greengenes2 checkpoints applied no
confidence thresholds to about 98 to 100% of reads, because their OOD gate
replaced the requested thresholds with 0.0. With that fixed, their
tau-normalised confidences fall below the documented thresholds for most species
calls. The 26.10.0 checkpoints replace them:

| model | how it was made |
| --- | --- |
| V1-V9, V5-V7 | the v1.0.0 networks with tau-normalisation (tau 0.5) undone exactly |
| V4, V3-V4 | retrained on the same reference and recipe without tau-normalisation or OOD gate (seeds 1005 and 1006) |

Median species-level F-measure on the 13 Taxobench mock communities per region
at 0.5/0.7 is 0.961 (whole 16S), 0.940 (V5-V7), 0.957 (V4) and 0.836 (V3-V4),
with median species abstention of 4.1%, 4.3%, 5.3% and 19.5%. See
[CHANGELOG.md](CHANGELOG.md) for the comparison with earlier releases.
