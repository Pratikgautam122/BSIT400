# ML Integration Selector: Project Overview

This project uses Machine Learning (ML) to dynamically select the optimal numerical integration method for multi-dimensional mathematical functions. The goal is to choose the solver that minimizes integration error under a fixed function-evaluation budget.

---

## Project Structure and Workflow

The project consists of a set of modular Python files that orchestrate the pipeline from data generation to model evaluation:

```
bsit400/
├── run_project.py          # The main orchestration script
├── clean_workspace.py      # Cleans up files from previous iterations
├── integration_methods.py  # Custom multidimensional solvers implemented from scratch
├── dataset_generator.py    # Generates synthetic functions and extracts features
├── train_model.py          # Trains ML models and plots findings
├── project_report.md       # Comprehensive mathematical and empirical report
└── figures/                # Output directory containing visual analysis plots
```

---

## How It Works

The workflow executes in four distinct phases:

### 1. Workspace Cleaning
The pipeline begins by running the [clean](file:///Users/pratikgautam/Documents/bsit400/clean_workspace.py#L27) function inside [clean_workspace.py](file:///Users/pratikgautam/Documents/bsit400/clean_workspace.py). This cleans the environment of artifacts from previous projects.

### 2. Synthetic Dataset Generation
To train a machine learning model, we need functions with *known exact integrals*.
- In [dataset_generator.py](file:///Users/pratikgautam/Documents/bsit400/dataset_generator.py), we define the [SyntheticFunction](file:///Users/pratikgautam/Documents/bsit400/dataset_generator.py#L140) class which constructs functions in $d$-dimensions ($d \in [1, 5]$) using a combination of separable (product) or additive (sum) 1D component functions.
- The component functions include polynomials, exponentials, trigonometric forms, absolute values, step functions, and singular functions. Because these components have analytical integrals, the exact integral of the resulting synthetic function can be computed analytically.
- We generate a dataset of 600 functions using [generate_dataset](file:///Users/pratikgautam/Documents/bsit400/dataset_generator.py#L394).

### 3. Feature Extraction & Labeling
For each synthetic function, we extract features and assign the optimal solver label:
- **Numerical Feature Extraction:** The [extract_features](file:///Users/pratikgautam/Documents/bsit400/dataset_generator.py#L283) function probes the function as a black box (evaluating it at 500 random sample points). It computes 18 features capturing shape, range, estimated first/second derivatives, spikiness, oscillation counts, boundary behavior, and potential singularities.
- **Labeling:** We evaluate four custom integration algorithms from [integration_methods.py](file:///Users/pratikgautam/Documents/bsit400/integration_methods.py) on the function using a target budget of $\approx 1024$ evaluations:
  1. [trapezoidal_nd](file:///Users/pratikgautam/Documents/bsit400/integration_methods.py#L88)
  2. [simpson_nd](file:///Users/pratikgautam/Documents/bsit400/integration_methods.py#L115)
  3. [romberg_nd](file:///Users/pratikgautam/Documents/bsit400/integration_methods.py#L150)
  4. [monte_carlo_nd](file:///Users/pratikgautam/Documents/bsit400/integration_methods.py#L200)
- The algorithm that achieves the *lowest absolute error* compared to the exact analytical integral is assigned as the training label (Class 0, 1, 2, or 3).

### 4. Model Training & Evaluation
We pass the extracted features and labels to [train_and_evaluate](file:///Users/pratikgautam/Documents/bsit400/train_model.py#L21) in [train_model.py](file:///Users/pratikgautam/Documents/bsit400/train_model.py):
- The dataset is split into training (75%) and testing (25%) sets.
- We train a **Decision Tree Classifier** and a **Random Forest Classifier** to predict the optimal integration method.
- The Decision Tree rules are printed to show high-interpretability splits.
- The trained Random Forest classifier achieves an accuracy of **55.33%** (vs. 25% random baseline) on unseen functions.
- The script automatically plots and saves the following visualizations to the [figures](file:///Users/pratikgautam/Documents/bsit400/figures) directory:
  - `feature_importances.png`: Identifies which features (such as function dimension and derivative spikes) matter most.
  - `confusion_matrix.png`: Illustrates the classification performance per solver.
  - `decision_boundary.png`: Visualizes the transition between solver regimes based on dimension and function ruggedness.

---

## How to Run

To run the entire pipeline, execute the orchestration script:

```bash
python run_project.py
```

This will run the clean utility, generate the dataset, train/evaluate the classifiers, save evaluation metrics to [ml_integration_results.json](file:///Users/pratikgautam/Documents/bsit400/ml_integration_results.json), and generate plots in the [figures](file:///Users/pratikgautam/Documents/bsit400/figures) folder.

For a detailed analysis of the mathematical formulations and empirical findings, refer to the [project_report.md](file:///Users/pratikgautam/Documents/bsit400/project_report.md).
