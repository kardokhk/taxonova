# Taxonova

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/taxonova.svg)](https://pypi.org/project/taxonova/)

**Taxonova** is a deep learning-based taxonomic classification system designed for high-accuracy species-level classification of microbial 16S rRNA amplicon sequences. Built on hierarchical neural networks, Taxonova achieves strong species-level precision across variable sequence lengths (250-4900 bp) and multiple hypervariable regions.

## Overview

Accurate taxonomic assignment is fundamental to microbial ecology and microbiome research. Taxonova addresses critical challenges in amplicon-based taxonomic classification, including:

- **Data imbalance** across taxonomic levels (domain to species)
- **Variable sequence lengths** from short (250 bp) to long (4900 bp) amplicons
- **Hierarchical consistency** in taxonomic predictions
- **Species-level resolution** in complex microbial communities

Taxonova employs a hierarchical few-shot learning framework with attention mechanisms to capture taxonomic relationships and achieve state-of-the-art classification performance, particularly at the species level where traditional methods often struggle.

## Access Options

### 🌐 Web Interface (Recommended for Most Users)

Use Taxonova directly through our hosted web application:

**[https://huggingface.co/spaces/claessonlab/Taxonova](https://huggingface.co/spaces/claessonlab/Taxonova)**

- No installation required
- User-friendly interface
- Suitable for single analyses and exploratory work
- Pre-configured with optimized models

### 💻 Command-Line Installation

For high-throughput analysis, custom model training, or integration into bioinformatics pipelines:

**Install via PyPI:**
```bash
pip install taxonova
```

**PyPI Repository:** [https://pypi.org/project/taxonova](https://pypi.org/project/taxonova)

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pre-trained Models](#pre-trained-models)
- [Usage](#usage)
  - [Classification](#classification)
  - [Training Custom Models](#training-custom-models)
- [Input Format](#input-format)
- [Output Format](#output-format)
- [Performance](#performance)
- [Citation](#citation)
- [License](#license)
- [Support](#support)

## Key Features

### Methodological Advances

- **Hierarchical Prototypical Classification**: Few-shot learning architecture specifically optimized for taxonomic hierarchies
- **Attention-Based Context Integration**: Multi-head attention mechanisms capture taxonomic relationships across classification levels
- **Balanced Hierarchical Loss**: Custom loss function addresses class imbalance from domain to species level
- **Taxonomic Consistency Enforcement**: Ensures predictions respect phylogenetic relationships

### Practical Capabilities

- **Broad Sequence Compatibility**: Supports amplicons from 250-4900 bp across multiple hypervariable regions (V1-V9, V3-V4, V4)
- **Pre-trained Models**: Ready-to-use models trained on GreenGenes2 and GTDB databases
- **GPU Acceleration**: CUDA support for efficient large-scale classification and training
- **Flexible Confidence Thresholding**: Taxonomic-level-specific confidence thresholds for controlled precision
- **Smart Training**: Automated two-stage training with early stopping for optimal generalization

## Installation

### Prerequisites

- Python ≥ 3.8
- 2 GB RAM minimum
- CUDA-compatible GPU (optional, recommended for training)


### Standard Installation

Install Taxonova and its dependencies from PyPI:

```bash
pip install taxonova
```

### Download Pre-trained Models

After installation, download pre-trained classification models:

```bash
# Interactive model selection (recommended)
taxonova-setup

# Or specify models explicitly
taxonova download-models --models gg2_v3v4 gg2_v4
```

Models are stored in `~/.taxonova/models/` by default. You can customize the location using the `TAXONOVA_MODELS_DIR` environment variable.

### Development Installation

For contributing to Taxonova or using the latest development version:

```bash
git clone https://github.com/kardokhk/taxonova.git
cd taxonova
pip install -e .
```

## Quick Start

### Classify sequences using a pre-trained model:

```bash
taxonova classification \
    --input sequences.fasta \
    --model_input gg2_v3v4_model.pt \
    --output predictions.tsv \
    --verbose
```

**Note:** If models are in the default location (`~/.taxonova/models/`), use only the model filename. Taxonova will automatically locate it.

### Example output:

```
sequence_id	domain	phylum	class	order	family	genus	species	domain_conf	phylum_conf	...
seq_001	Bacteria	Firmicutes	Bacilli	Lactobacillales	Lactobacillaceae	Lactobacillus	Lactobacillus_acidophilus	0.99	0.97	...
```

## Pre-trained Models

Taxonova provides models trained on curated reference databases for different 16S rRNA regions:

| Model ID | Database | Region | Sequences | Use Case |
|----------|----------|--------|-----------|----------|
| `gg2_v1v9` | GreenGenes2 | V1-V9 | ~400K | Near full-length 16S |
| `gg2_v3v4` | GreenGenes2 | V3-V4 | ~400K | Standard Illumina MiSeq |
| `gg2_v4` | GreenGenes2 | V4 | ~400K | Short amplicon studies |
| `gg2_rrn` | GreenGenes2 | Full rRNA | ~400K | Long-read sequencing |
| `gtdb_rrn` | GTDB | Full rRNA | ~300K | GTDB taxonomy framework |

### Download Models

**Interactive selection:**
```bash
taxonova download-models --interactive
```

**Specific models:**
```bash
taxonova download-models --models gg2_v3v4 gg2_v4
```

**All models:**
```bash
taxonova download-models --models all
```

**List available models:**
```bash
taxonova download-models --list
```

## Usage

### Classification

Classify unknown amplicon sequences using pre-trained models.

#### Basic Classification

```bash
taxonova classification \
    --input input_sequences.fasta \
    --model_input gg2_v3v4_model.pt \
    --output results.tsv \
    --verbose
```

#### Parameters

- `--input`: Input FASTA file containing sequences to classify (required)
- `--model_input`: Path to trained model file (required)
- `--output`: Output file path for classification results (required)
- `--confidence_threshold`: Global confidence threshold (default: 0.5)
- `--species_confidence`: Species-level confidence threshold (default: 0.5)
- `--genus_confidence`: Genus-level confidence threshold (default: 0.5)
- `--enforce_taxonomy`: Enforce hierarchical consistency (default: True)
- `--verbose`: Display detailed progress information

#### Advanced Classification with Custom Thresholds

```bash
taxonova classification \
    --input sequences.fasta \
    --model_input gg2_v3v4_model.pt \
    --output results.tsv \
    --domain_threshold 0.95 \
    --phylum_threshold 0.90 \
    --class_threshold 0.85 \
    --order_threshold 0.80 \
    --family_threshold 0.75 \
    --genus_threshold 0.70 \
    --species_threshold 0.65 \
    --verbose
```

Higher thresholds increase precision at the cost of potentially more unassigned classifications.

### Training Custom Models

Train Taxonova models on custom reference databases for specialized applications.

#### Standard Training

```bash
taxonova training \
    --train_db reference_database.fasta \
    --model_output custom_model.pt \
    --batch_size 32 \
    --epochs 30 \
    --learning_rate 0.01 \
    --kmer-length 8 \
    --verbose
```

#### Smart Full Data Training (Recommended)

For optimal performance, use the smart training mode which employs a two-stage approach:

1. **Scouting phase**: Uses 20% validation split to determine optimal number of epochs
2. **Full training phase**: Trains on 100% of data for the optimal number of epochs

```bash
taxonova training \
    --train_db reference_database.fasta \
    --model_output custom_model.pt \
    --batch_size 32 \
    --learning_rate 0.01 \
    --kmer-length 8 \
    --smart_full_data \
    --verbose
```

#### Training Parameters

- `--train_db`: Reference database in FASTA format with taxonomic headers (required)
- `--model_output`: Output path for trained model (required)
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 30; auto-determined with `--smart_full_data`)
- `--learning_rate`: Learning rate for optimization (default: 0.01)
- `--kmer-length`: K-mer size for sequence encoding (default: 8)
- `--smart_full_data`: Enable smart two-stage training
- `--optimize_metric`: Optimization metric: `species_f1` or `val_loss` (default: species_f1)

#### Custom Taxonomic Level Weights

Adjust the importance of different taxonomic levels during training:

```bash
taxonova training \
    --train_db database.fasta \
    --model_output model.pt \
    --domain_weight 0.2 \
    --phylum_weight 0.3 \
    --class_weight 0.4 \
    --order_weight 0.6 \
    --family_weight 0.8 \
    --genus_weight 1.0 \
    --species_weight 1.5 \
    --smart_full_data \
    --verbose
```

Higher weights prioritize accuracy at specific taxonomic levels.

## Input Format

Taxonova accepts FASTA-formatted files with taxonomic information in sequence headers.

### For Classification

Standard FASTA format:

```
>sequence_id_001
ATGCATGCATGCATGC...
>sequence_id_002
GCTAGCTAGCTAGCTA...
```

### For Training

FASTA format with taxonomic annotation in headers:

```
>seq001 d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Lactobacillaceae;g__Lactobacillus;s__Lactobacillus_acidophilus
ATGCATGCATGCATGC...
>seq002 d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia_coli
GCTAGCTAGCTAGCTA...
```

**Taxonomic header format:**
- Semicolon-separated taxonomic ranks
- Prefixes: `d__` (domain), `p__` (phylum), `c__` (class), `o__` (order), `f__` (family), `g__` (genus), `s__` (species)
- Compatible with GTDB and GreenGenes2 taxonomy formats

## Output Format

Classification results are saved in tab-delimited format:

| Column | Description |
|--------|-------------|
| `sequence_id` | Input sequence identifier |
| `domain` | Predicted domain |
| `phylum` | Predicted phylum |
| `class` | Predicted class |
| `order` | Predicted order |
| `family` | Predicted family |
| `genus` | Predicted genus |
| `species` | Predicted species |
| `domain_conf` | Domain confidence score (0-1) |
| `phylum_conf` | Phylum confidence score (0-1) |
| `class_conf` | Class confidence score (0-1) |
| `order_conf` | Order confidence score (0-1) |
| `family_conf` | Family confidence score (0-1) |
| `genus_conf` | Genus confidence score (0-1) |
| `species_conf` | Species confidence score (0-1) |

Unassigned taxonomic levels (below confidence threshold) are marked as `Unassigned`.

## Performance

Taxonova demonstrates competitive performance on benchmark datasets:

- **Species-level F1 Score**: >0.85 on balanced test sets
- **Genus-level Accuracy**: >0.92
- **Hierarchical Consistency**: 100% (enforced by architecture)
- **Processing Speed**: ~1000-5000 sequences/second (GPU-dependent)

Performance varies based on:
- Sequence length and quality
- Database coverage and taxonomic diversity
- Confidence threshold settings
- Hardware specifications

## Citation

If you use Taxonova in your research, please cite:

```
Taxonova: Deep Learning-based Hierarchical Taxonomic Classification for Bacterial Amplicon Sequences
Kardokh Kaka Bra
2025
GitHub: https://github.com/kardokhk/taxonova
PyPI: https://pypi.org/project/taxonova
```

## License

Taxonova is released under the MIT License. See [LICENSE](LICENSE) for details.

## Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/kardokhk/taxonova/issues)
- **Documentation**: See [README](https://github.com/kardokhk/taxonova#readme) and inline help (`taxonova --help`)
- **Web Interface**: [https://huggingface.co/spaces/claessonlab/Taxonova](https://huggingface.co/spaces/claessonlab/Taxonova)

## Acknowledgments

Taxonova builds upon established taxonomic frameworks and reference databases:
- **GreenGenes2**: McDonald et al., 2023
- **GTDB**: Parks et al., 2022

Development supported by the Claesson Lab.

---

**Developed by:** Kardokh Kaka Bra  
**Contact:** kardokhkaka@gmail.com  
**Repository:** https://github.com/kardokhk/taxonova

