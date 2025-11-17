# Taxonova Quick Reference

Quick reference for common Taxonova commands and workflows.

## Installation

```bash
# Install from PyPI
pip install taxonova

# Download models (interactive)
taxonova-setup

# Or download models manually
taxonova download-models --interactive
```

## Model Management

```bash
# List available models
taxonova download-models --list

# Download specific models
taxonova download-models --models gg2_v3v4 gg2_v4

# Download all models
taxonova download-models --models all

# Force re-download
taxonova download-models --models gg2_v3v4 --force
```

## Classification

```bash
# Basic classification
taxonova classification \
    --input sequences.fasta \
    --model_input gg2_v3v4_model.pt \
    --output results.tsv

# With custom confidence thresholds
taxonova classification \
    --input sequences.fasta \
    --model_input gg2_v3v4_model.pt \
    --output results.tsv \
    --species_confidence 0.7 \
    --genus_confidence 0.6 \
    --verbose

# Using model from custom location
taxonova classification \
    --input sequences.fasta \
    --model_input /path/to/custom_model.pt \
    --output results.tsv
```

## Training

```bash
# Basic training
taxonova training \
    --train_db database.fasta \
    --model_output my_model.pt \
    --batch_size 32 \
    --epochs 30 \
    --kmer-length 8

# Smart full data training (recommended)
taxonova training \
    --train_db database.fasta \
    --model_output my_model.pt \
    --batch_size 32 \
    --learning_rate 0.01 \
    --kmer-length 8 \
    --smart_full_data \
    --verbose

# Custom taxonomic weights
taxonova training \
    --train_db database.fasta \
    --model_output my_model.pt \
    --domain_weight 0.2 \
    --phylum_weight 0.3 \
    --class_weight 0.4 \
    --order_weight 0.6 \
    --family_weight 0.8 \
    --genus_weight 1.0 \
    --species_weight 1.5
```

## Environment Variables

```bash
# Set custom models directory
export TAXONOVA_MODELS_DIR=/data/taxonova/models

# Set custom config directory  
export TAXONOVA_CONFIG_DIR=/data/taxonova/config

# Force CPU usage (no GPU)
CUDA_VISIBLE_DEVICES="" taxonova classification ...
```

## File Locations

- **Models**: `~/.taxonova/models/` (default)
- **Config**: `~/.taxonova/` (default)
- **Logs**: `output/` (in current directory)

## Input Format

### Training Database (FASTA)
```
>seq1|d:Bacteria;p:Proteobacteria;c:Gammaproteobacteria;o:Enterobacterales;f:Enterobacteriaceae;g:Escherichia;s:Escherichia_coli
ATCGATCGATCG...
```

### Query Sequences (FASTA)
```
>seq1
ATCGATCGATCG...
>seq2
GCTAGCTAGCTA...
```

## Output Format (TSV)

```
sequence_id    domain    phylum    class    order    family    genus    species    domain_confidence    ...
seq1          Bacteria  Proteo... Gamma... Enter... Enter... Esche... E_coli   0.99                 ...
```

## Common Options

### Global Options
- `--verbose`: Show detailed progress
- `--show-resource-usage`: Display memory and time metrics
- `--help`: Show help message

### Training Options
- `--train_db`: Training database (required)
- `--model_output`: Output model path (required)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 30)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--kmer-length`: K-mer length (default: 8)
- `--smart_full_data`: Enable smart training

### Classification Options
- `--input`: Input sequences (required)
- `--model_input`: Model file (required)
- `--output`: Output file (required)
- `--species_confidence`: Species threshold (default: 0.6)
- `--genus_confidence`: Genus threshold (default: 0.5)
- `--family_confidence`: Family threshold (default: 0.4)

## Troubleshooting

### Model not found
```bash
# Check available models
taxonova download-models --list

# Download required model
taxonova download-models --models gg2_v3v4
```

### Out of memory
```bash
# Reduce batch size
taxonova training --batch_size 16 ...

# Or use CPU only
CUDA_VISIBLE_DEVICES="" taxonova training ...
```

### CUDA errors
```bash
# Force CPU usage
CUDA_VISIBLE_DEVICES="" taxonova classification ...
```

## Getting Help

```bash
# General help
taxonova --help

# Mode-specific help
taxonova classification --help
taxonova training --help
taxonova download-models --help
```

## Examples

### Complete Workflow

```bash
# 1. Install
pip install taxonova

# 2. Download models
taxonova download-models --models gg2_v3v4

# 3. Classify sequences
taxonova classification \
    --input my_sequences.fasta \
    --model_input gg2_v3v4_model.pt \
    --output classifications.tsv \
    --verbose

# 4. View results
head -n 10 classifications.tsv
```

### Custom Training

```bash
# 1. Prepare your database (FASTA with taxonomic headers)

# 2. Train model
taxonova training \
    --train_db my_database.fasta \
    --model_output my_custom_model.pt \
    --smart_full_data \
    --verbose

# 3. Use your custom model
taxonova classification \
    --input sequences.fasta \
    --model_input my_custom_model.pt \
    --output results.tsv
```

## Version Info

```bash
# Check version
taxonova --version

# Check Python version
python --version

# Check installed packages
pip show taxonova
```

## More Information

- Full documentation: `README.md`
- Publishing guide: `PUBLISHING.md`
- Changelog: `CHANGELOG.md`
- GitHub: https://github.com/YOUR_USERNAME/taxonova
- PyPI: https://pypi.org/project/taxonova/
