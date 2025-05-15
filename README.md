# Taxonova 🧬

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep learning system for accurate hierarchical taxonomic classification of biological sequences. The system uses state-of-the-art deep learning techniques to classify sequences across multiple taxonomic levels (domain, phylum, class, order, family, genus, species).

## Table of Contents
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [Training Mode](#1-training-mode-training)
  - [Cross-Validation Mode](#2-cross-validation-mode-cv)
  - [Classification Mode](#3-classification-mode-classification)
  - [Tuning Mode](#4-tuning-mode-tuning)
  - [Weight Tuning Mode](#5-weight-tuning-mode-weight-tuning)
- [Input Data Format](#input-data-format)
- [Output Files](#output-files)
- [Advanced Configuration](#advanced-configuration)
- [Hardware Requirements](#hardware-requirements)
- [Citation](#citation)
- [License](#license)

## Key Features

- **Hierarchical Classification**: Simultaneously predicts multiple taxonomic levels
- **High Accuracy**: Uses advanced deep learning architectures optimized for taxonomic classification
- **Cross-Validation**: Robust model evaluation with k-fold cross-validation
- **Hyperparameter Tuning**: Automated optimization of model parameters
- **Detailed Metrics**: Comprehensive performance metrics and visualizations
- **GPU Support**: CUDA acceleration for faster training and prediction

## Prerequisites

- Python 3.8
- CUDA-capable GPU (recommended)
- 16GB RAM (minimum)

## Installation

1. Create a conda environment:
```bash
conda create -n taxonova python=3.8
conda activate taxonova
```

2. Clone the repository and navigate to the project directory

3. Make the script executable:
```bash
chmod +x bin/taxonova
```

4. Install requirements:
```bash
pip install -r requirements.txt
```

5. Install the package in development mode:
```bash
pip install -e .
```

6. Verify the installation:
```bash
taxonova --help
```

## Usage Guide

### 1. Training Mode (training)

Train and save a new model:

```bash
taxonova training \
    --train_db database/db.fasta \
    --epochs 50 \
    --batch_size 32 \
    --model_output models/my_model.pt \
    --full_data \
    --learning_rate 0.1 \
    --verbose
```

Features:
- Full model training with early stopping
- Model checkpoint saving
- Training progress visualization
- Resource usage monitoring

### 2. Cross-Validation Mode (cv)

Evaluate model performance with k-fold cross-validation:

```bash
taxonova cv \
    --train_db database/db.fasta \
    --epochs 10 \
    --folds 10 \
    --batch_size 32 \
    --verbose
```

Outputs:
- Performance metrics for each fold
- Statistical analysis across folds
- Visualization plots (loss curves, accuracy metrics, performance distribution)

### 3. Classification Mode (classification)

Make predictions on new sequences:

```bash
taxonova classification \
    --input input/test_query_input.fasta \
    --model_input models/my_model.pt \
    --output predictions.tsv
```

Output format (TSV file):
```
sequence_id   domain   domain_confidence   phylum   phylum_confidence   ...
seq1         Bact...  0.9234             Prot...  0.8756             ...
seq2         Bact...  0.8845             Firm...  0.7634             ...
```

### 4. Tuning Mode (tuning)

Optimize model hyperparameters:

```bash
taxonova tuning \
    --train_db database/ref_db_tiny.fasta \
    --epochs 10 \
    --folds 10
```

Features:
- Random search over hyperparameter space
- Cross-validation for each parameter set
- Performance metrics visualization
- Comprehensive hyperparameter optimization

### 5. Weight Tuning Mode (weight-tuning)

Specifically tune hierarchical classification weights to optimize F1 scores:

```bash
taxonova weight-tuning \
    --train_db database/ref_db_tiny.fasta \
    --epochs 5 \
    --folds 5 \
    --iterations 30 \
    --family_min 0.2 --family_max 1.5 \
    --genus_min 0.2 --genus_max 1.5 \
    --species_min 0.5 --species_max 2.0
```

For grid search instead of random search:

```bash
taxonova weight-tuning \
    --train_db database/ref_db_tiny.fasta \
    --epochs 5 \
    --folds 5 \
    --grid_search \
    --grid_points 5
```

Features:
- Specialized for optimizing the weights between taxonomic levels
- Find optimal weight balance between family, genus, and species levels
- Supports both random search and grid search
- Cross-validation for robust weight evaluation
- Detailed metrics showing F1 scores for each level
- Saves top weight configurations for reference

## Input Data Format

### Training Data
FASTA format with taxonomy in headers:
```
>sequence_id domain phylum class order family genus species
ATGCTAGCTAG...
```

### Prediction Data
FASTA format with sequence IDs:
```
>sequence_id
ATGCTAGCTAG...
```

## Output Files

### 1. Model Files (Training Mode)
- `*.pt`: Model weights and architecture
- `*.taxonomy.pt`: Taxonomy information and configuration

### 2. Evaluation Results (CV Mode)
- Detailed metrics for each fold
- Statistical summaries
- Performance visualization plots

### 3. Classification Results
- TSV file with predictions and confidence scores
- Optional visualization plots

### 4. Tuning Results
- Best hyperparameter combinations
- Performance metrics for each trial
- Parameter importance visualization
- Convergence plots

## Advanced Configuration

### Training Parameters
- `--early_stopping`: Enable early stopping (default: True)
- `--warmup_fraction`: Learning rate warmup period (default: 0.1)
- `--weight_decay`: L2 regularization strength (default: 1e-5)
- `--learning_rate`: Initial learning rate (default: 5e-5)

### Output Control
- `--verbose`: Show detailed progress in console
- `--show-resource-usage`: Display memory and time statistics

## Hardware Requirements

- **Recommended**: CUDA-capable GPU
- **Minimum RAM**: 16GB
- CPU-only mode available (slower performance)

## Citation

If you use this software in your research, please cite:
[Citation information to be added]

## License

[License information to be added]
