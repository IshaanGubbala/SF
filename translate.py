#!/usr/bin/env python3

"""
predict_eeg_mlp.py

Usage example:
    python translate.py \
        --input_csv extracted_features.csv \
        --model_dir trained_models \
        --output_csv average_predictions.csv
"""

import argparse
import os
import pandas as pd
import joblib
import numpy as np

# Make sure these match whatever you used during training & feature extraction
FEATURE_NAMES = [
    "Delta", "Theta1", "Theta2", "Alpha1", "Alpha2", "Beta1", "Beta2", "Gamma1", "Gamma2",
    "Alpha_Ratio", "Theta_Ratio",
    "Shannon_Entropy",
    "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity",
    "Spatial_Complexity"
]

def main():
    parser = argparse.ArgumentParser(
        description="Run MLP predictions on a CSV of EEG features."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the CSV file containing extracted EEG features."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="trained_models",
        help="Directory where MLP model and scaler are saved."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="predictions.csv",
        help="Path to output CSV with predictions and confidence."
    )
    args = parser.parse_args()

    input_csv = args.input_csv
    model_dir = args.model_dir
    output_csv = args.output_csv

    # ------------------ Load the MLP model and scaler ------------------------
    mlp_model_path = os.path.join(model_dir, "mlp_fold_4.joblib")
    mlp_scaler_path = os.path.join(model_dir, "mlp_scaler_fold_4.joblib")

    if not os.path.exists(mlp_model_path):
        raise FileNotFoundError(f"Could not find MLP model file: {mlp_model_path}")
    if not os.path.exists(mlp_scaler_path):
        raise FileNotFoundError(f"Could not find MLP scaler file: {mlp_scaler_path}")

    print(f"[INFO] Loading MLP model from {mlp_model_path}")
    mlp_model = joblib.load(mlp_model_path)
    print(f"[INFO] Loading scaler from {mlp_scaler_path}")
    mlp_scaler = joblib.load(mlp_scaler_path)

    # ------------------ Load the features CSV -------------------------------
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"[INFO] Reading features from {input_csv}")
    df = pd.read_csv(input_csv)

    # Check if all FEATURE_NAMES are in the CSV columns
    missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    # Reorder the DataFrame columns to match the training feature order exactly
    df_ordered = df[FEATURE_NAMES]

    # ------------------ Run Predictions -------------------------------------
    # Scale the feature matrix
    X = df_ordered.values  # shape: (n_samples, n_features)
    X_scaled = mlp_scaler.transform(X)

    # Predict classes
    predictions = mlp_model.predict(X_scaled)
    # For binary classification: probability of the positive class
    probabilities = mlp_model.predict_proba(X_scaled)[:, 1]

    # ------------------ Save predictions ------------------------------------
    # Add the predictions and confidence (probabilities) to the original df
    df["prediction"] = predictions
    df["confidence"] = probabilities

    print(f"[INFO] Saving predictions to {output_csv}")
    df.to_csv(output_csv, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()