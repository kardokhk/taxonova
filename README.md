# Taxonova

[![PyPI](https://img.shields.io/pypi/v/taxonova.svg)](https://pypi.org/project/taxonova/)
[![Paper](https://img.shields.io/badge/paper-PLoS%20Comput%20Biol-blue.svg)](https://doi.org/10.1371/journal.pcbi.1014912)
[![Web app](https://img.shields.io/badge/web%20app-Hugging%20Face-orange.svg)](https://huggingface.co/spaces/claessonlab/Taxonova)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/kardokhk/taxonova/blob/main/LICENSE)
[![Tests](https://github.com/kardokhk/taxonova/actions/workflows/ci.yml/badge.svg)](https://github.com/kardokhk/taxonova/actions/workflows/ci.yml)

Taxonova assigns 16S rRNA and rRNA operon amplicon sequences to a lineage from
domain to species, with a confidence at each rank, and stops at the deepest rank
it is confident about.

## Classify your sequences

**No installation:** upload a FASTA file to the
[web app](https://huggingface.co/spaces/claessonlab/Taxonova).

**Command line** (Python 3.10 or later):

```bash
pip install taxonova
taxonova classify reads.fasta --model v4
```

The model is downloaded on first use (about 240 MB, once). Results are written
to `reads.taxonova.tsv`.

## Models

| id | amplicon | reference |
| --- | --- | --- |
| `v4` | 16S V4 | Greengenes2 |
| `v3v4` | 16S V3-V4 | Greengenes2 |
| `v5v7` | 16S V5-V7 | Greengenes2 |
| `16s` | whole 16S gene (V1-V9) | Greengenes2 |
| `operon` | rRNA operon (16S-ITS-23S) | GROND |

Species names follow GTDB. SILVA and RDP models are in preparation.
Run `taxonova models` to see which models are downloaded.

## Output

One row per sequence: a label and a confidence for each rank (columns for
phylum to family not shown here).

```
sequence_id   domain    domain_confidence  ...  genus       genus_confidence  species                 species_confidence
f1eb3726...   Bacteria  1.0000             ...  Prevotella  0.9988            Prevotella pectinovora  0.9988
```

A rank below its confidence threshold, and every rank below it, reads
`Unclassified`.

## Options you may want

```bash
taxonova classify reads.fasta -m v3v4 -o results.tsv   # model and output file
taxonova classify reads.fasta -m v4 --confidence 0.9   # stricter (default 0.8)
taxonova classify reads.fasta -m v4 --threads 8        # more CPU threads
taxonova classify reads.fasta -m v4 --genus-threshold 0.6   # set one rank
```

`--confidence` sets the species threshold, and the other ranks scale with it.
From Python: `taxonova.classify("reads.fasta", model="v4")` returns a pandas
DataFrame.

## Links

- [Documentation](https://github.com/kardokhk/taxonova/tree/main/docs): [models](https://github.com/kardokhk/taxonova/blob/main/docs/models.md), [output and thresholds](https://github.com/kardokhk/taxonova/blob/main/docs/output.md), [Python](https://github.com/kardokhk/taxonova/blob/main/docs/python-api.md), [training your own model](https://github.com/kardokhk/taxonova/blob/main/docs/training.md)
- [Changelog](https://github.com/kardokhk/taxonova/blob/main/CHANGELOG.md)
- [Paper](https://doi.org/10.1371/journal.pcbi.1014912) and the benchmark used in it, [TaxoBench](https://github.com/kardokhk/TaxoBench)
- [Report a problem or ask for a model](https://github.com/kardokhk/taxonova/issues/new/choose)

## Citation

If you use Taxonova, please cite:

> Kakabra K, Patel S, Temko A, Claesson MJ. Taxonova: deep learning enables
> fast and confidence-aware microbiome species-level amplicon classification.
> PLoS Comput Biol. 2026;22(10):e1014912. doi:10.1371/journal.pcbi.1014912

Taxonova is developed in the Claesson Lab, University College Cork and APC
Microbiome Ireland, and released under the
[MIT licence](https://github.com/kardokhk/taxonova/blob/main/LICENSE).
