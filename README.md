# Taxonova

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/taxonova.svg)](https://pypi.org/project/taxonova/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Taxonova is a deep-learning classifier for the taxonomic assignment of microbial
rRNA amplicons. It takes 16S rRNA amplicon reads, or full rRNA operons, and gives
you a taxonomy down to species level when the sequence carries enough signal to
support it.

The thing that makes it different from a stack of independent classifiers is that
the taxonomy is built into the network. Instead of predicting seven ranks
separately and hoping they agree, Taxonova runs a domain-to-species cascade where
each rank is predicted in the context of its parents through cross-level attention.
A genus call is informed by the order and family the model already settled on, a
species call is informed by the genus, and so on down the tree. That keeps the
output internally consistent and lets the model lean on the easier, higher ranks
to anchor the harder, finer ones.

It also knows when to stop. Short hypervariable regions (a 250 bp V4 read, say)
often just do not contain enough information to separate two close relatives. When
that happens Taxonova abstains at the species level and returns a genus-level call
instead of guessing. Think of it as "genus known, species uncertain" rather than a
confident-but-wrong species label. You can tune how cautious it is with confidence
thresholds.

Species labels are anchored in a whole-genome reference. The four amplicon models
use Greengenes2, which takes its nomenclature from GTDB; the full-operon model uses
GTDB directly. So the species names you get back are the genome-based names, not
16S-only operational guesses.

## Why you might want it

- Hierarchical, species-level calls that stay internally consistent across ranks.
- Calibrated abstention: it returns a genus call instead of a wrong species call
  when the region cannot resolve the species.
- Works across a wide span of amplicons, from short single-region reads to the
  full rRNA operon.
- Light on resources. It runs comfortably on a laptop; no GPU needed for
  classification.

## Performance

These numbers are from the manuscript (currently in revision). They are the
species-level results across a broad benchmark of amplicon regions and lengths.

- Species-level F-measure of 0.76 to 1.00 (median 0.94) on short-read amplicons,
  and 0.82 to 0.99 (median 0.97) on long-read, across amplicons spanning roughly
  250 bp to 4,900 bp.
- Roughly an order of magnitude fewer false-positive species calls than the
  commonly used tools we compared against.
- 5- to 40-fold lower abundance-estimation error, depending on the region.

The abstention behaviour is a big part of why the false-positive rate is low: the
model declines to make a species call it cannot back up, rather than inflating its
apparent coverage.

## Runtime and footprint

Classification runs on a single CPU core at roughly 100 sequences per second,
inside a roughly constant 0.82 GB memory footprint regardless of how many
sequences you throw at it. No GPU is required. A GPU only helps if you are
training your own model from scratch, and even then it is optional.

## Installation

Development install, straight from this repository:

```bash
git clone https://github.com/kardokhk/taxonova.git
cd taxonova
pip install -e .
```

It is also on PyPI:

```bash
pip install taxonova
```

https://pypi.org/project/taxonova/

And if you would rather not install anything at all, there is a hosted web app:

https://huggingface.co/spaces/claessonlab/Taxonova

## Pretrained models

There are five pretrained models, one per amplicon region or operon. Pick the one
that matches how your reads were generated.

| Model file | Region | Reference |
| --- | --- | --- |
| `gg2_v4_mrgroups_model.pt` | V4 | Greengenes2 |
| `gg2_v3v4_mrgroups_model.pt` | V3-V4 | Greengenes2 |
| `gg2_v5v7_mrgroups_model.pt` | V5-V7 | Greengenes2 |
| `gg2_v1v9_mrgroups_model.pt` | V1-V9 (near-full 16S) | Greengenes2 |
| `gtdb_rrn_mrgroups_model.pt` | RRN (full rRNA operon) | GTDB |

Each checkpoint is around 235 to 246 MB, which is too big to sit in the git repo,
so they are attached to the GitHub Release instead.

The easy way is the built-in downloader. After installing the package you can run:

```bash
# interactive picker (also available straight after install as `taxonova-setup`)
taxonova-setup

# or grab specific models non-interactively
taxonova download-models --models gg2_v4
taxonova download-models --models all

# see what is available and what is already downloaded
taxonova download-models --list
```

The model keys are `gg2_v4`, `gg2_v3v4`, `gg2_v5v7`, `gg2_v1v9`, and `gtdb_rrn`.
Downloads land in `~/.taxonova/models/` by default; set the `TAXONOVA_MODELS_DIR`
environment variable to store them somewhere else. Once a model is in that
directory you can pass just its filename (for example
`gg2_v4_mrgroups_model.pt`) to `--model_input` and Taxonova will find it.

The manual alternative still works: grab the `.pt` file you need directly from the
Releases page and point `--model_input` at it.

https://github.com/kardokhk/taxonova/releases

The confidence calibration (tau-norm rescaling) is already baked into these
checkpoints, so there is nothing extra to run. More detail in
[MODELS.md](MODELS.md).

## Usage

Taxonova has five modes: `training`, `classification`, `calibrate`, `fit-ood`, and
`download-models`. For day-to-day use you only need `download-models` (once) and
`classification`.

### Classification

```bash
taxonova classification \
    --input reads.fasta \
    --model_input gg2_v4_mrgroups_model.pt \
    --output predictions.tsv
```

The flags that matter:

- `--input` (required): your query sequences, in FASTA.
- `--model_input` (required): a pretrained `.pt` checkpoint.
- `--output` (required): where to write the TSV.
- `--confidence_threshold` (default 0.5): the species-level confidence cutoff.
  Calls below this are reported as `Unclassified` at species.
- Per-rank overrides, all default to None (which falls back to 0.5 internally):
  `--domain_threshold`, `--phylum_threshold`, `--class_threshold`,
  `--order_threshold`, `--family_threshold`, `--genus_threshold`. The species cutoff
  is controlled by `--confidence_threshold`.
- `--cpu_threads`: number of CPU threads to use.
- `--verbose`: chatty progress output.

You can pass either a full path to a `.pt` checkpoint or, if you fetched it with
`taxonova download-models`, just the filename (Taxonova resolves it from
`~/.taxonova/models/`, or `TAXONOVA_MODELS_DIR` if you set it).

### Input and output

Input is plain FASTA. Headers can be just an ID, or an ID followed by a
Greengenes/GTDB-style lineage string (`d__...; p__...; ...; s__...`) if you happen
to have one, though for classification the lineage is ignored.

Output is a tab-separated table, one row per input sequence. The first column is
`sequence_id`, then for each of the seven ranks there is a label column and a
confidence column:

```
sequence_id  domain  domain_confidence  phylum  phylum_confidence  ...  species  species_confidence
```

A couple of things to know about the confidences. They are forced to be
monotonic down the tree, so a species confidence can never exceed its genus
confidence, which can never exceed its family confidence, and so on, the way
probabilities in a nested hierarchy have to behave. And uncertainty propagates
downward: if a rank falls below its threshold it is written as `Unclassified`, and
every rank below it is `Unclassified` too. That is the abstention mechanism in
practice. A read that resolves cleanly to genus but not species will show real
labels through `genus` and `Unclassified` at `species`.

### Training your own model

If you want to train on your own reference set, the simplest path is
`--smart_full_data`. It does a two-stage run: first it scouts on a validation
split to find a good number of epochs (with early stopping), then it retrains on
100% of your data for that many epochs, so you do not waste reference sequences on
a permanent holdout.

```bash
taxonova training \
    --train_db reference.fasta \
    --model_output my_model.pt \
    --smart_full_data \
    --verbose
```

The training FASTA should have lineage strings in the headers
(`d__...; ...; s__...`). Useful knobs include `--epochs`, `--batch_size`,
`--learning_rate`, `--kmer-length`, the per-rank loss weights
(`--domain_weight` ... `--species_weight`), and length augmentation
(`--length_aug_p`, `--length_aug_min`, `--length_aug_max`) for making a model
robust to short reads. Run `taxonova training --help` for the full list.

After training you can fit post-hoc calibration with `taxonova calibrate` (per-rank
temperature scaling) and, if you care about out-of-distribution inputs, a per-input
OOD threshold with `taxonova fit-ood`. The shipped pretrained models already have
calibration applied, so you only need these if you are rolling your own.

## Quick start

The fastest way to see it work, with no downloads, is the offline smoke test using
the tiny example files in `examples/`. Train a small throwaway model on the bundled
reference, then classify the bundled queries:

```bash
# 1) train a tiny model on the example reference (a minute or two on CPU)
taxonova training \
    --train_db examples/tiny_db.fasta \
    --model_output /tmp/tiny_model.pt \
    --epochs 3 --batch_size 16 --learning_rate 0.01

# 2) classify the example queries with it
taxonova classification \
    --input examples/tiny_test.fasta \
    --model_input /tmp/tiny_model.pt \
    --output /tmp/tiny_predictions.tsv
```

That model is too small to be accurate; it just proves the pipeline runs. For real
work, download a pretrained model and classify with that:

```bash
# fetch the V4 model into ~/.taxonova/models/
taxonova download-models --models gg2_v4

taxonova classification \
    --input reads.fasta \
    --model_input gg2_v4_mrgroups_model.pt \
    --output predictions.tsv
```

There is a slightly fuller walkthrough in [QUICKSTART.md](QUICKSTART.md).

## Citation

If Taxonova is useful in your work, please cite the manuscript:

> Kardokh Kaka Bra. Taxonova: hierarchical deep-learning taxonomic classification
> of microbial rRNA amplicons. Manuscript in revision.

A formal citation with DOI will be added here once the paper is published.

## License

MIT. See [LICENSE](LICENSE).

## Contact

Kardokh Kaka Bra, kardokhkaka@gmail.com

## Acknowledgments

Taxonova builds directly on the genome-based taxonomies it is trained against:
Greengenes2 (McDonald et al., 2023) for the amplicon models and GTDB (Parks et al.)
for the full-operon model. Developed in the Claesson Lab.
