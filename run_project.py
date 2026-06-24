"""
run_project.py
--------------
Orchestrates the entire project execution:
  1. Cleans the workspace of old ODE files (runs clean_workspace.py).
  2. Generates a dataset of synthetic functions with exact integrals and features.
  3. Preprocesses data and trains the Decision Tree and Random Forest classifiers.
  4. Saves evaluation metrics, console outputs, and plots in the figures/ folder.
"""

import sys
import json
import time
import os

from clean_workspace import clean
from dataset_generator import generate_dataset
from train_model import train_and_evaluate

def main():
    start_time = time.time()
    
    # 1. Clean workspace (safety check)
    clean()
    
    # 2. Generate data
    print("--- STEP 1: Generating Dataset ---")
    dataset_start = time.time()
    # 600 functions to ensure at least 500+ valid samples after skips
    feats, labels, methods_names = generate_dataset(n_samples=600, target_evals=1024, seed=42)
    dataset_elapsed = time.time() - dataset_start
    print(f"Dataset generated in {dataset_elapsed:.2f} seconds.\n")
    
    # 3. Train models and generate plots
    print("--- STEP 2: Training and Evaluating ML Models ---")
    train_start = time.time()
    metrics = train_and_evaluate(feats, labels, methods_names, figures_dir="figures")
    train_elapsed = time.time() - train_start
    print(f"ML models trained and evaluated in {train_elapsed:.2f} seconds.\n")
    
    # 4. Save results to JSON
    overall_elapsed = time.time() - start_time
    summary = {
        'total_functions_generated': len(labels),
        'dataset_generation_time_s': dataset_elapsed,
        'model_training_time_s': train_elapsed,
        'total_project_execution_time_s': overall_elapsed,
        'decision_tree_accuracy': metrics['accuracy_dt'],
        'random_forest_accuracy': metrics['accuracy_rf'],
        'feature_names': metrics['feature_names'],
        'rf_feature_importances': metrics['importances'],
        'methods_mapping': methods_names
    }
    
    with open("ml_integration_results.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("Saved ml_integration_results.json")
    print(f"\nProject executed successfully in {overall_elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
