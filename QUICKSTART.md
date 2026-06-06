# Quickstart

Five minutes, start to finish. We will first run a no-download smoke test on the
tiny example files, then do one real classification with a pretrained model.

## Install

```bash
git clone https://github.com/kardokhk/taxonova.git
cd taxonova
pip install -e .
```

Or just `pip install taxonova` if you do not need the example files.

## 1. Offline smoke test (no downloads)

The repo ships a tiny reference (`examples/tiny_db.fasta`) and a tiny query set
(`examples/tiny_test.fasta`). Train a small throwaway model and classify with it.
This only checks that the pipeline runs end to end; the model is far too small to
be accurate.

```bash
# train a tiny model on the bundled reference (a minute or two on CPU)
taxonova training \
    --train_db examples/tiny_db.fasta \
    --model_output /tmp/tiny_model.pt \
    --epochs 3 --batch_size 16 --learning_rate 0.01

# classify the bundled queries with it
taxonova classification \
    --input examples/tiny_test.fasta \
    --model_input /tmp/tiny_model.pt \
    --output /tmp/tiny_predictions.tsv

# take a look
column -t -s $'\t' /tmp/tiny_predictions.tsv | less -S
```

If `predictions.tsv` has one row per input sequence with a label and a confidence
for each rank, everything is wired up correctly.

## 2. A real classification

For actual work, download a pretrained model from the Releases page and use that.
Choose the model that matches your amplicon region (see [MODELS.md](MODELS.md)).

https://github.com/kardokhk/taxonova/releases

```bash
# example: V4 reads
taxonova classification \
    --input reads.fasta \
    --model_input gg2_v4_mrgroups_model.pt \
    --output predictions.tsv
```

That is it. The calibration is already inside the checkpoint, so there is no extra
setup step.

## Reading the output

The TSV has `sequence_id` followed by a label and a confidence column for each of
the seven ranks (domain through species). Where the model is not confident enough
at a rank, you will see `Unclassified` there and at every rank below it. That is
the model abstaining rather than guessing. A read that resolves to genus but not
species is the common and expected case for short regions.

## Tuning how cautious it is

`--confidence_threshold` (default 0.5) sets the species cutoff. Raise it for fewer,
higher-confidence species calls; lower it to let more through. You can also set
per-rank cutoffs from domain through genus with `--domain_threshold` ...
`--genus_threshold`.

```bash
taxonova classification \
    --input reads.fasta \
    --model_input gg2_v4_mrgroups_model.pt \
    --output predictions.tsv \
    --confidence_threshold 0.7
```

Want more detail on training, modes, and the architecture? See the
[README](README.md).
