# Pretrained models

Taxonova ships five pretrained models, one for each amplicon region or operon.
Pick the one that matches how your sequences were generated, download it, and pass
it to `--model_input`. That is the whole workflow.

| Model file | Use it for | Reference taxonomy |
| --- | --- | --- |
| `gg2_v4_mrgroups_model.pt` | V4 amplicons | Greengenes2 |
| `gg2_v3v4_mrgroups_model.pt` | V3-V4 amplicons | Greengenes2 |
| `gg2_v5v7_mrgroups_model.pt` | V5-V7 amplicons | Greengenes2 |
| `gg2_v1v9_mrgroups_model.pt` | V1-V9 / near-full-length 16S | Greengenes2 |
| `gtdb_rrn_mrgroups_model.pt` | RRN, the full rRNA operon | GTDB |

The four Greengenes2 models take their species nomenclature from GTDB by way of
Greengenes2; the RRN model uses GTDB directly.

## Where they live

Each checkpoint is roughly 235 to 246 MB, too large to keep inside the git repo, so
they are attached to the GitHub Release rather than committed. Download them from
the Releases page:

https://github.com/kardokhk/taxonova/releases

## Downloading and using one

Grab the checkpoint for your region (here, V4):

```bash
# from the Releases page, e.g. with curl/wget once you have the asset URL
wget -O gg2_v4_mrgroups_model.pt <release-asset-url-for-gg2_v4_mrgroups_model.pt>
```

Then point classification at it:

```bash
taxonova classification \
    --input reads.fasta \
    --model_input gg2_v4_mrgroups_model.pt \
    --output predictions.tsv
```

## Calibration is already baked in

Each checkpoint already has its tau-norm confidence calibration applied. You do not
need to run `taxonova calibrate` on these; the confidences and the abstention
behaviour are ready to use as-is. Calibration only matters if you train your own
model from your own reference set.
