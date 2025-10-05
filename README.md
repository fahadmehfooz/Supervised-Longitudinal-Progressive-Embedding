# SLOPE: Self-supervised Longitudinal Progression Embedding for Alzheimer's Disease

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

## Overview

SLOPE (Self-supervised Longitudinal Progression Embedding) is a novel machine learning framework for modeling the continuous progression trajectory of Alzheimer's disease (AD). Unlike traditional approaches that classify patients into discrete diagnostic groups, SLOPE learns a continuous representation that captures the full spectrum of disease progression while preserving temporal consistency across longitudinal follow-up visits.

### Key Innovation

The core innovation of SLOPE lies in its **direction loss** mechanism, which enforces biologically plausible progression trajectories by ensuring that amyloid accumulation either remains stable or increases over time—reflecting the irreversible nature of AD pathology. This addresses a critical limitation in existing models that often produce inconsistent predictions across multiple visits for the same patient.

## Features

- **Continuous Disease Modeling**: Maps the entire AD spectrum as a continuous trajectory rather than discrete categories
- **Temporal Consistency**: Enforces monotonic progression using longitudinal context from follow-up visits
- **Self-supervised Learning**: Trains without requiring diagnostic labels, using only temporal relationships
- **Generalization**: Applies learned trajectory to unseen patients for progression tracking
- **Visual Interpretation**: Generates intuitive 2D progression trajectories for clinical monitoring

## Dataset

The model is validated on the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** dataset:
- **961 subjects** across 2,023 visits
- **68 cortical regions** of interest from amyloid PET imaging
- Diagnostic categories: Cognitively Normal (CN), Early MCI, Late MCI, Alzheimer's Disease
- **526 subjects** with longitudinal follow-up data for temporal validation

## Installation

### Prerequisites

```bash
pip install torch torchvision
pip install numpy pandas scipy scikit-learn
pip install umap-learn matplotlib seaborn
pip install torchviz
pip install jupyter
```

### Additional R Dependencies (for Slingshot)
```r
install.packages("BiocManager")
BiocManager::install("slingshot")
```

### Clone Repository

```bash
git clone https://github.com/JW-Yan/SLOPE.git
cd SLOPE
```

## Usage

### Basic Training

```python
import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
import numpy as np

# Set random seed for reproducibility
set_global_seed(42)

# Load and prepare your data
# Assuming you have X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
# and train_data_identifiers, test_data_identifiers DataFrames

# Define model architecture
dims = [68, 32, 16, 8, 16, 32, 68]  # input_dim -> bottleneck -> output_dim
model = AE(dims=dims, activation="tanh", dropout=0.1, batch_norm=True)

# Define loss criterion
criterion = nn.MSELoss()

# Train the model
trained_model, umap_model, train_embeddings, train_encodings = train_AE(
    model=model,
    train_data=train_data,
    test_data=test_data,
    X_train_tensor=X_train_tensor,
    y_train_tensor=y_train_tensor,
    X_test_tensor=X_test_tensor,
    y_test_tensor=y_test_tensor,
    train_data_identifiers=train_data_identifiers,
    test_data_identifiers=test_data_identifiers,
    criterion=criterion,
    optimizer_class=Adam,
    lr=0.001,
    epochs=500,
    lambda_reconstruction=1.0,
    lambda_directional=0.5,  # Key parameter for direction loss
    directional_loss_enabled=True
)

# Get test embeddings
test_embeddings, test_encodings = test_AE(
    model=trained_model,
    X_test_tensor=X_test_tensor,
    y_test_tensor=y_test_tensor,
    test_data_identifiers=test_data_identifiers,
    umap_model=umap_model
)
```

### Training with Different Loss Weights

```python
# For datasets with more temporal noise, increase directional loss weight
trained_model, umap_model, train_embeddings, train_encodings = train_AE(
    model=model,
    # ... other parameters
    lambda_reconstruction=1.0,
    lambda_directional=1.0,  # Higher weight for direction loss
    directional_loss_enabled=True
)

# For datasets with fewer longitudinal samples, reduce directional loss
trained_model, umap_model, train_embeddings, train_encodings = train_AE(
    model=model,
    # ... other parameters
    lambda_reconstruction=1.0,
    lambda_directional=0.2,  # Lower weight for direction loss
    directional_loss_enabled=True
)
```

## Model Architecture

### Autoencoder Structure

The SLOPE model uses a symmetric autoencoder architecture:

```python
# Example architecture for 68-dimensional input
dims = [68, 32, 16, 8, 16, 32, 68]
#       ^input  ^bottleneck  ^output

model = AE(
    dims=dims,
    activation="tanh",     # Activation function
    dropout=0.1,          # Dropout rate for regularization
    batch_norm=True       # Batch normalization
)
```

### Loss Components

The total loss combines reconstruction and directional losses:

```
L_total = λ₁ * L_reconstruction + λ₂ * L_directional
```

Where:
- **L_reconstruction**: MSE loss between input and reconstructed output
- **L_directional**: Cosine similarity loss between consecutive visit progression vectors

## Classification Performance

Downstream classification results (CN vs AD) on held-out test subjects:

| Method | Feature Type | Accuracy | F1-Score | Balanced Accuracy | ROC-AUC |
|--------|--------------|----------|----------|-------------------|---------|
| **SLOPE + Logistic Regression** | Embeddings | **0.863** | **0.841** | **0.858** | **0.858** |
| **SLOPE + Elastic Net** | Embeddings | **0.863** | **0.841** | **0.858** | **0.858** |
| **SLOPE + MLP** | Embeddings | **0.863** | **0.841** | **0.858** | **0.858** |
| Autoencoder + Logistic Regression | Embeddings | 0.843 | 0.814 | 0.836 | 0.836 |
| Autoencoder + Elastic Net | Embeddings | 0.843 | 0.814 | 0.836 | 0.836 |
| Autoencoder + MLP | Embeddings | 0.843 | 0.814 | 0.836 | 0.836 |
| Original + Logistic Regression | Raw SUVR | 0.804 | 0.762 | 0.794 | 0.794 |
| Original + Elastic Net | Raw SUVR | 0.814 | 0.776 | 0.805 | 0.805 |
| Original + MLP | Raw SUVR | 0.814 | 0.765 | 0.801 | 0.801 |

## Results

### Classification Performance Highlights
- **Consistent 86.3% accuracy** across all classifiers when using SLOPE embeddings
- **2-8% improvement** in F1-score compared to autoencoder embeddings
- **6-10% improvement** in F1-score compared to original SUVR features
- **Superior balanced accuracy** indicating better performance across class imbalance

### Biological Validity
SLOPE identifies biologically meaningful progression patterns:
- **Early changes**: Precuneus and posterior cingulate cortex (default mode network)
- **Late changes**: Precentral, postcentral, and lateral occipital cortices
- Consistent with established AD pathology literature

### Generalization
The learned trajectory successfully generalizes to held-out test subjects, maintaining consistent group separation and temporal ordering.

## Key Training Parameters

```python
# Recommended hyperparameters
epochs = 1550
dims = [68, 286, 20, 286, 68]
activation = nn.ReLU
learning_rate = 2.4430162614261403e-05
criterion = nn.MSELoss()
optimizer = torch.optim.Adam
lambda_reconstruction = 0.40921304830970556
lambda_directional = 1
```

## Utility Functions

The framework provides several utility functions:

```python
# Set deterministic behavior
set_global_seed(42)

# Save trained models
save_pickle_object("slope_model.pkl", trained_model)
save_pickle_object("umap_model.pkl", umap_model)

# Initialize model weights deterministically
initialize_weights(model, seed=42)
```

## Data Requirements

Your data should include:
- **Feature tensors**: `X_train_tensor`, `X_test_tensor` (PyTorch tensors)
- **Label tensors**: `y_train_tensor`, `y_test_tensor` (PyTorch tensors)
- **Identifier DataFrames**: Must contain `RID` (subject ID) and `EXAMDATE` columns for temporal ordering
- **Data DataFrames**: Must contain `DXGrp` column for diagnostic groups

## Project Structure

```
SLOPE/
├── README.md
├── requirements.txt
├── slope/
│   ├── __init__.py
│   ├── model.py              # Core SLOPE implementation
│   ├── losses.py             # Direction loss and reconstruction loss
│   ├── utils.py              # Data processing utilities
│   └── visualization.py     # Plotting functions
├── data/
│   ├── preprocessing/        # ADNI data preprocessing scripts
│   └── sample_data/         # Example datasets
├── notebooks/
│   ├── SLOPE_Tutorial.ipynb # Complete tutorial
│   ├── Evaluation.ipynb     # Model evaluation
│   └── Visualization.ipynb  # Results visualization
├── results/
│   ├── embeddings/          # Saved model embeddings
│   ├── figures/             # Generated plots
│   └── metrics/             # Evaluation results
└── tests/
    ├── test_model.py        # Unit tests
    └── test_losses.py       # Loss function tests
```

## Reproducibility

To reproduce results from the paper:

```bash
cd notebooks/
jupyter notebook SLOPE_Paper_Results.ipynb
```

All experiments use fixed random seeds and the same 80/20 train-test split as reported in the manuscript.

## Citation

If you use SLOPE in your research, please cite:

```bibtex
@article{mehfooz2024slope,
  title={Learning a Continuous Progression Trajectory of Alzheimer's disease for Personalized Tracking},
  author={Mehfooz, Fahad and Tong, Mingzhao and Zhang, Shu and Wang, Yipei and Fang, Shiaofen and Saykin, Andrew J. and Wang, Xiaoqian and Yan, Jingwen},
  journal={[Journal Name]},
  year={2024}
}
```

## Data Availability

- **Source code**: Available on [GitHub](https://github.com/JW-Yan/SLOPE)
- **ADNI data**: Available through [LONI IDA](https://ida.loni.usc.edu/) upon request

## Support and Contributing

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join the community discussion for questions and ideas
- **Contributing**: See CONTRIBUTING.md for guidelines on submitting pull requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This research was supported by NIH grants R01 AG081951, U19 AG074879, U01 AG068057, and NSF 2345235, 1942394. Data collection and sharing was funded by ADNI.

---

**Note**: This is a research tool intended for academic and research purposes. It should not be used for clinical diagnosis without proper validation and regulatory approval.
