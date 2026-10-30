# Training your own model

Use this when no pretrained model fits your data: another primer pair, another
reference database, or a curated in-house reference. A GPU is recommended for
training on a reference of realistic size; classification with the result runs
on a CPU.

## Reference format

A FASTA file in which each header carries the lineage, ranks separated by
semicolons with the GTDB-style prefixes `d__`, `p__`, `c__`, `o__`, `f__`,
`g__` and `s__`:

```
>seq1 d__Bacteria; p__Bacillota; c__Bacilli; o__Lactobacillales; f__Lactobacillaceae; g__Lactobacillus; s__Lactobacillus crispatus
TACGTAGGTGGCAAGCGTTGTCCGGATTTATTGGGCGTAAAGCGAGCGCAGG...
```

Trim the reference to the amplicon you sequence (the same primers), so that the
model sees the same region in training as in your data. An empty species
(`s__`) is allowed.

## Train

```bash
taxonova training --train_db reference.fasta --model_output my_model.pt --smart_full_data
```

`--smart_full_data` first trains with a validation split to find the number of
epochs, then retrains on all sequences for that many epochs. Other options
include `--epochs`, `--batch_size`, `--learning_rate`, `--kmer-length`
(default 8), `--filter_incomplete_lineages` and `--require_species`. Run
`taxonova training --help` for the full list.

Then classify with the file:

```bash
taxonova classify reads.fasta --model my_model.pt
```

If you copy `my_model.pt` into the models directory
(`$TAXONOVA_MODELS_DIR`, else `~/.taxonova/models`), `--model my_model.pt`
finds it from any working directory.

## Calibrate (optional)

`calibrate` fits one temperature per rank on a held-out part of the reference
and stores it in the checkpoint, so that confidences better match the observed
accuracy:

```bash
taxonova calibrate --model_input my_model.pt --train_db reference.fasta \
    --model_output my_model.calibrated.pt --report_json calibration.json
```

The held-out part is set by `--val_split` (default 0.2) and `--seed`
(default 42).

## Out-of-distribution gate (optional)

`fit-ood` fits a threshold on an out-of-distribution score (by default the
maximum phylum probability) and stores it in the checkpoint. A sequence whose
score marks it as unlike the reference is then gated at the higher of the
requested threshold and the gate's own threshold at each rank; the gate never
lowers a threshold.

```bash
taxonova fit-ood --model_input my_model.pt --train_db reference.fasta \
    --model_output my_model.ood.pt --id_quantile 0.99
```

`--id_quantile 0.99` keeps 99% of held-out in-distribution sequences as
in-distribution. The pretrained models carry no such gate (see
[models.md](models.md)).

## A quick check without a GPU

The repository ships a small reference and query set. Training on them checks
that the installation works; the model is far too small to be useful.

```bash
git clone https://github.com/kardokhk/taxonova.git && cd taxonova
taxonova training --train_db examples/tiny_db.fasta --model_output tiny.pt \
    --epochs 2 --batch_size 16 --learning_rate 0.01
taxonova classify examples/tiny_test.fasta --model tiny.pt
```

This takes about a minute on a CPU.
