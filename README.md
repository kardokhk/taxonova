# Taxonova

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/taxonova.svg)](https://pypi.org/project/taxonova/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Web server](https://img.shields.io/badge/web%20server-Hugging%20Face-orange.svg)](https://huggingface.co/spaces/claessonlab/Taxonova)

Taxonova is a deep learning classifier for the taxonomic assignment of microbial
ribosomal RNA amplicon sequences, from single variable regions of the 16S rRNA gene
to the full rRNA operon. For each sequence it reports a lineage from domain to
species together with a confidence at each rank. When the confidence at a rank is
below its threshold, Taxonova reports the lineage only down to the rank above, for
example a genus without a species, rather than assigning an unsupported label. The
thresholds therefore set the balance between assigning more sequences and making
fewer incorrect assignments.

Species names follow the Genome Taxonomy Database (GTDB): the 16S rRNA models are
trained on Greengenes2, which takes its nomenclature from GTDB, and the full-operon
model on GROND with GTDB taxonomy.

## Web server

**The simplest way to use Taxonova is the web server, which needs no installation:**

**https://huggingface.co/spaces/claessonlab/Taxonova**

Upload a FASTA file or paste sequences, select the amplicon region and, if needed, a
confidence threshold. The web server screens each sequence with an rRNA gene filter
before classification and returns the same results as the command-line tool. For
large datasets, or to train a model on your own reference, install the package as
described below.

## Performance

In the accompanying manuscript, evaluated on physical and biome-simulated mock
communities against seven published classifiers, Taxonova reached species-level
F-measures of 0.76 to 1.00 (median 0.94) on short-read and 0.82 to 0.99 (median
0.97) on long-read amplicons of 250 bp to 4,900 bp. Those figures were obtained
with the v1.0.0 checkpoints and the original Taxobench scoring rules. With the
26.10.0 checkpoints and the documented thresholds, median species-level
F-measures over the 13 mock communities per region are 0.836 to 0.961 at 0.5/0.7
(CHANGELOG.md). Under the revised Taxobench scoring rules (v2), which do not count
an abstention as a predicted taxon, Taxonova no longer has the lowest species
false-positive rate of the eight classifiers. It classifies about 100 sequences
per second on a single CPU core with a peak memory use of 0.82 GB; no GPU is used
for classification.

## Installation

```bash
pip install taxonova
```

or, from this repository:

```bash
git clone https://github.com/kardokhk/taxonova.git
cd taxonova
pip install -e .
```

## Pretrained models

| Model file | Region | Reference |
| --- | --- | --- |
| `gg2_v4_mrgroups_model_26.10.pt` | V4 | Greengenes2 |
| `gg2_v3v4_mrgroups_model_26.10.pt` | V3-V4 | Greengenes2 |
| `gg2_v5v7_mrgroups_model_26.10.pt` | V5-V7 | Greengenes2 |
| `gg2_v1v9_mrgroups_model_26.10.pt` | whole 16S (V1-V9) | Greengenes2 |
| `gtdb_rrn_mrgroups_model.pt` | full rRNA operon (16S-ITS-23S) | GROND, GTDB taxonomy |

The checkpoints (about 240 MB each) are attached to the
[GitHub releases](https://github.com/kardokhk/taxonova/releases) (Greengenes2:
v26.10.0; RRN: v1.0.0) and can be downloaded with the built-in tool:

```bash
taxonova download-models --list            # available and downloaded models
taxonova download-models --models gg2_v4   # one model; keys: gg2_v4, gg2_v3v4, gg2_v5v7, gg2_v1v9, gtdb_rrn
taxonova download-models --models all      # all five
```

Models are stored in `~/.taxonova/models/`, or in the directory given by the
`TAXONOVA_MODELS_DIR` environment variable, and can then be referred to by file
name. The downloader verifies each file's checksums.

**Upgrading from 26.9.x or earlier:** releases 1.0.0 to 26.9.2 applied no
confidence thresholds to the Greengenes2 checkpoints, and 26.10.0 replaces those
checkpoints. Download the new ones and rerun earlier Greengenes2 analyses; see
[CHANGELOG.md](CHANGELOG.md) for how output changes and [MODELS.md](MODELS.md)
for how the checkpoints were made.

## Usage

```bash
taxonova classification \
    --input reads.fasta \
    --model_input gg2_v4_mrgroups_model_26.10.pt \
    --output predictions.tsv
```

| Option | Meaning |
| --- | --- |
| `--input` | query sequences in FASTA format |
| `--model_input` | pretrained checkpoint (path or file name) |
| `--output` | output table (TSV) |
| `--confidence_threshold` | species threshold, default 0.8; the thresholds of the other ranks scale with it |
| `--domain_threshold` ... `--genus_threshold` | set the threshold of a single rank explicitly |
| `--cpu_threads` | number of CPU threads |

The default thresholds are 0.5 for domain to family, 0.7 for genus and 0.8 for
species. Because the same threshold does not give the same abstention rate in every
amplicon region or with every reference, we recommend choosing the threshold that
gives the abstention rate a study requires on a held-out subset of the reference.

The output table has one row per sequence: `sequence_id`, then a label and a
confidence for each of the seven ranks. The confidence at each rank is capped at
that of the rank above. A rank whose confidence is below its threshold, and every
rank below it, is reported as `Unclassified`.

Other modes are `training` (train a model on your own reference; `--smart_full_data`
selects the number of epochs on a validation split and then retrains on all
sequences), `calibrate` (temperature scaling) and `fit-ood` (out-of-distribution
threshold). Run `taxonova <mode> --help` for all options, and see
[QUICKSTART.md](QUICKSTART.md) for a short worked example.

## Benchmark

The evaluation framework used in the manuscript, Taxobench, is available at
https://github.com/kardokhk/TaxoBench.

## Citation

> Kakabra K, Patel S, Temko A, Claesson MJ. Taxonova: deep learning enables fast
> and accurate microbiome species-level amplicon classification. Manuscript in
> revision.

A full citation will be added on publication.

## License

MIT; see [LICENSE](LICENSE).

## Contact

Kardokh Kakabra, kardokhkaka@gmail.com. Developed in the Claesson Lab, University
College Cork and APC Microbiome Ireland.
