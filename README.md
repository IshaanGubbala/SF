# Downstream EEG Feature Training Pipeline

This project provides a comprehensive pipeline for training and evaluating machine learning models using pre-extracted EEG features. It includes implementations of both a traditional Multi-Layer Perceptron (MLP) and a quantum-inspired model, **QSupFullNet**, designed to enhance classification performance on EEG data.

## Table of Contents

- [Introduction](#introduction)
- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
  - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
  - [QSupFullNet](#qsupfullnet)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Electroencephalography (EEG) signals are complex and often require sophisticated processing pipelines to extract meaningful patterns. This project aims to facilitate the downstream analysis of EEG data by providing a structured approach to load pre-extracted features, train machine learning models, and evaluate their performance. The inclusion of the QSupFullNet model introduces a novel quantum-inspired architecture tailored for EEG data classification.

## Pipeline Overview

The pipeline encompasses the following steps:

1. **Loading Pre-Extracted Features**: Handcrafted and Graph Neural Network (GNN) features are loaded from specified directories.
2. **Data Preparation**: Features are combined, labels are assigned based on participant information, and data is split into training and validation sets.
3. **Model Training**: Both MLP and QSupFullNet models are trained using the prepared datasets.
4. **Evaluation**: Models are evaluated using cross-validation, with performance metrics logged and visualized.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/IshaanGubbala/SF.git
   cd SF
   ```

2. **Install required dependencies**:

   Ensure you have Python 3.8 or later installed. Install dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, if you use conda:

   ```bash
   conda env create -f environment.yml
   conda activate eeg-pipeline
   ```

3. **Set up directories**:

   Create necessary directories for processed features and logs:

   ```bash
   mkdir -p processed_features/handcrafted
   mkdir -p processed_features/gnn
   mkdir -p processed_features/channels
   mkdir -p plots
   mkdir -p logs
   ```

4. **Update paths**:

   Modify the paths in the script to point to your local data files:

   ```python
   PARTICIPANTS_FILE_DS004504 = "/path/to/ds004504/participants.tsv"
   PARTICIPANTS_FILE_DS003800 = "/path/to/ds003800/participants.tsv"
   ```

## Usage

After setting up the environment and directories, run the main script to execute the pipeline:

```bash
python main.py
```

This will:

- Load the pre-extracted EEG features.
- Train both MLP and QSupFullNet models.
- Evaluate their performance using cross-validation.
- Generate and save plots for loss curves and ROC curves in the `plots` directory.
- Log training details in the `logs` directory.

## Model Architectures

### Multi-Layer Perceptron (MLP)

The MLP model serves as a baseline classifier. It is implemented using scikit-learn's `MLPClassifier` with the following configuration:

- **Hidden Layers**: Two layers with 26 and 14 neurons, respectively.
- **Activation Function**: Logistic sigmoid.
- **Solver**: Adam optimizer.
- **Regularization (Alpha)**: 0.275.
- **Training Approach**: Partial fit to accommodate incremental learning.

### QSupFullNet

`QSupFullNet` is a quantum-inspired neural network designed to handle the superposition of multiple hypotheses, enhancing its capability to model uncertainty and complex patterns in EEG data. Key features include:

- **Wavefunction Networks**: Multiple sub-networks generate complex-valued representations of the input.
- **ArcBell Activation**: A Gaussian-like activation function defined as `exp(-z^2)`.
- **Partial Norm**: Normalization technique to control the magnitude of wavefunctions.
- **Self-Modulation**: Optional gating mechanism for dynamic feature modulation.
- **Top-K Aggregation**: Focuses on the most significant features by selecting the top-K components.

This architecture is implemented using PyTorch and leverages the `torch_geometric` library for graph-based operations.

## Training and Evaluation

The training process involves cross-validation to assess model performance robustly:

1. **Cross-Validation Setup**: Stratified K-Folds (default is 3 folds) to maintain class distribution.
2. **Data Augmentation**: Synthetic Minority Over-sampling Technique (SMOTE) is applied to address class imbalance.
3. **Standardization**: Features are standardized to have zero mean and unit variance.
4. **Training**: Both models are trained on each fold, with training and validation losses logged.
5. **Evaluation Metrics**: Accuracy, confusion matrix, classification report, and ROC curves are generated to evaluate model performance.

## Results

Upon completion of the training and evaluation pipeline:

- **Logs**: Detailed training and validation logs are saved in the `logs` directory.
- **Plots**: Loss curves and ROC curves for each fold and model are saved in the `plots` directory.
- **Performance Metrics**: Overall accuracy, confusion matrices, and 