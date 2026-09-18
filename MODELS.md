# Pretrained models

Taxonova ships five pretrained models, one for each amplicon region or operon.
Pick the one that matches how your sequences were generated, download it, and pass
it to `--model_input`. That is the whole workflow.

| Model file | Use it for | Reference taxonomy |
| --- | --- | --- |
| `gg2_v4_mrgroups_model.pt` | V4 amplicons | Greengenes2 |
| `gg2_v3v4_mrgroups_model.pt` | V3-V4 amplicons | Greengenes2 |
| `gg2_v5v7_mrgroups_model.pt` | V5-V7 amplicons | Greengenes2 |
| `gg2_v1v9_mrgroups_model.pt` | whole 16S (V1-V9) | Greengenes2 |
| `gtdb_rrn_mrgroups_model.pt` | RRN, the full rRNA operon | GROND, GTDB taxonomy |

The four Greengenes2 models take their species nomenclature from GTDB by way of
Greengenes2; the RRN model is trained on GROND, which uses GTDB taxonomy.

## Where they live

Each checkpoint is roughly 235 to 246 MB, too large to keep inside the git repo, so
they are attached to the GitHub Release rather than committed.

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

Models are stored in `~/.taxonova/models/` by default. Set the
`TAXONOVA_MODELS_DIR` environment variable to keep them somewhere else. Once a
model is in that directory you can pass just its filename to `--model_input` and
Taxonova will resolve it automatically:

```bash
taxonova classification \
    --input reads.fasta \
    --model_input gg2_v4_mrgroups_model.pt \
    --output predictions.tsv
```

You can still download the assets manually from the Releases page and pass a full
path to `--model_input` instead:

https://github.com/kardokhk/taxonova/releases

## Calibration is already baked in

Each checkpoint already has its tau-norm confidence calibration applied. You do not
need to run `taxonova calibrate` on these; the confidences and the abstention
behaviour are ready to use as-is. Calibration only matters if you train your own
model from your own reference set.
