# Downstream EEG Feature Training Pipeline

This project provides a comprehensive pipeline for training and evaluating machine learning models using pre-extracted EEG features. It includes implementations of both a traditional Multi-Layer Perceptron (MLP) and a quantum-inspired model, **QSupFullNet**, designed to enhance classification performance on EEG data.

## Table of Contents

- [Introduction](#introduction)
- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
  - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
  - [QSup: Quantum Superposition-Inspired Neural Network](#qsup-quantum-superposition-inspired-neural-network)
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

### QSup: Quantum Superposition-Inspired Neural Network

QSup is an innovative neural network architecture that integrates principles from quantum mechanics, specifically the concept of superposition, into traditional artificial intelligence models. This approach enables the network to evaluate multiple hypotheses concurrently, enhancing its ability to handle uncertainty and recognize complex patterns.

#### How QSup Works

QSup operates through a series of steps that mirror quantum mechanical principles:

1. **Input Representation**

   *Wave Guesses*: For each input, QSup generates multiple "wave guesses," each offering a different perspective by incorporating both real and imaginary components.

2. **Superposition Formation**

   *Combination*: These wave guesses are combined into a single superposed state, encapsulating all potential interpretations of the input.

3. **Measurement and Collapse**

   *Evaluation*: Upon processing, the superposed state is measured, causing a "collapse" into a definitive output based on the probabilities derived from the combined wave guesses.

4. **Learning and Adjustment**

   *Optimization*: The network adjusts its parameters to improve the accuracy of future predictions, refining the generation and combination of wave guesses.

#### Mathematical Formulation

To provide a concrete understanding, let's delve into the mathematical underpinnings of QSup:

1. **Input Representation**

   Each input $\( x \)$ is transformed into a complex-valued vector $\( \psi(x) \)$, where:

   $$\psi(x) = \sum_{i} \alpha_i(x) + i\,\beta_i(x)$$

   Here, $\( \alpha_i(x) \)$ and $\( \beta_i(x) \)$ represent the real and imaginary components of the input transformation, respectively.

2. **Superposition Formation**

   The network processes the input through layers of complex-valued weights $\( W \)$ and biases $\( b \)$:

   $$z = W\,\psi(x) + b$$

   An activation function $\( f \)$ (e.g., a complex variant of ReLU) is applied:

   $$\psi' = f(z)$$

   This results in a superposed state $\( \psi' \)$ that combines multiple interpretations of the input.

3. **Measurement and Collapse**

   The final output is obtained by measuring the probability distribution $\( P(y\mid x) \)$:

   $$P(y\mid x) = \frac{|\langle \phi_y \mid \psi' \rangle|^2}{\sum_{y'} |\langle \phi_{y'} \mid \psi' \rangle|^2}$$

   where $\( \phi_y \)$ denotes the basis state corresponding to output $\( y \)$.

4. **Learning and Adjustment**

   The network's parameters are optimized by minimizing a loss function $\( \mathcal{L} \)$, such as the cross-entropy between the predicted and true distributions:

   $$\mathcal{L} = -\sum_{x} \sum_{y} P_{\text{true}}(y\mid x) \log P(y\mid x)$$

   Gradient-based optimization techniques are employed to update the parameters:

   $$W \leftarrow W - \eta\,\frac{\partial \mathcal{L}}{\partial W}$$

   where $\( \eta \)$ is the learning rate.

#### Hyperparameters and Tuning

Hyperparameters in QSup are critical as they govern the behavior and performance of the network.

- **Number of Wave Guesses $(\( N \))$**: Determines how many potential interpretations are considered simultaneously.  
  *Tuning*: A higher $\( N \)$ may capture more complexity but increases computational load.

- **Learning Rate $(\( \eta \))$**: Controls the speed at which the network updates its parameters during training.  
  *Tuning*: A balance is necessary; too high can lead to instability, too low can slow convergence.

- **Superposition Depth $(\( D \))$**: Defines the number of layers through which wave guesses are combined before measurement.  
  *Tuning*: Greater depth can model more complex relationships but may also introduce overfitting.

- **Regularization Parameters $(\( \lambda \))$**: Prevent overfitting by penalizing overly complex models.  
  *Tuning*: Adjusting $\( \lambda \)$ helps maintain a balance between model complexity and generalization.

*Common Tuning Methods*:

- **Grid Search**: Explores a predefined set of hyperparameter values exhaustively.
- **Random Search**: Samples hyperparameter combinations randomly.
- **Bayesian Optimization**: Uses probabilistic models to predict promising hyperparameter settings.
- **Early Stopping**: Monitors performance on a validation set and halts training when improvements plateau.

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
- **Performance Metrics**: Overall accuracy, confusion matrices, and additional classification metrics are generated and reported.

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get involved.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

We thank all contributors and collaborators who have supported the development of this project.

---

*Note: The mathematical formulations provided are intended to offer insight into the underlying principles of the QSup model. For full implementation details, refer to the source code and accompanying documentation.*
