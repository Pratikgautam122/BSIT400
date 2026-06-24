"""
train_model.py
-------------
Processes function integration datasets, trains Decision Tree and Random Forest
classifiers to select the optimal integration method, evaluates model performance,
and generates rich aesthetic plots for analysis.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def train_and_evaluate(feats, labels, methods_names, figures_dir="figures"):
    """
    Trains classifiers, prints metrics, and saves evaluation figures.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Convert list of dicts to NumPy array
    feature_names = sorted(feats[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in feats])
    y = np.array(labels)
    
    # Print dataset distribution
    print("Dataset distribution:")
    for idx, name in enumerate(methods_names):
        count = np.sum(y == idx)
        print(f"  Class {idx} ({name:12s}): {count} samples ({count/len(y)*100:.1f}%)")
    print(f"Total samples: {len(y)}\n")
    
    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # 3. Train Decision Tree
    dt_clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
    dt_clf.fit(X_train, y_train)
    
    # Train Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_clf.fit(X_train, y_train)
    
    # 4. Evaluate Models
    y_pred_dt = dt_clf.predict(X_test)
    y_pred_rf = rf_clf.predict(X_test)
    
    acc_dt = np.mean(y_pred_dt == y_test)
    acc_rf = np.mean(y_pred_rf == y_test)
    
    print("--- Decision Tree Performance ---")
    print(f"Test Accuracy: {acc_dt*100:.2f}%")
    print(classification_report(y_test, y_pred_dt, target_names=methods_names, labels=np.arange(len(methods_names)), zero_division=0))
    
    print("--- Random Forest Performance ---")
    print(f"Test Accuracy: {acc_rf*100:.2f}%")
    print(classification_report(y_test, y_pred_rf, target_names=methods_names, labels=np.arange(len(methods_names)), zero_division=0))
    
    # Print Decision Tree Rules (high interpretability)
    print("--- Extracted Decision Tree Rules (First 3 levels) ---")
    tree_rules = export_text(dt_clf, feature_names=feature_names, max_depth=3)
    # Map class indices to names in the text printout
    for idx, name in enumerate(methods_names):
        tree_rules = tree_rules.replace(f"class: {idx}", f"class: {name}")
    print(tree_rules)
    
    # 5. Plot Feature Importances (Random Forest)
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6.5))
    # Curated gradient-like color palette
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(importances)))
    plt.barh(range(len(importances)), importances[indices], color=colors, edgecolor='none', height=0.6)
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices], fontsize=10)
    plt.xlabel("Gini Feature Importance", fontsize=12, fontweight='bold', labelpad=10)
    plt.title("Function Characteristics Determining Optimal Solver", fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig_imp_path = os.path.join(figures_dir, "feature_importances.png")
    plt.savefig(fig_imp_path, dpi=140)
    plt.close()
    print(f"Saved figure: {fig_imp_path}")
    
    # 6. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_rf, labels=np.arange(len(methods_names)))
    
    plt.figure(figsize=(7.5, 6.5))
    # Use a premium HSL-like blues/purples colormap
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Random Forest Confusion Matrix", fontsize=13, fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(methods_names))
    plt.xticks(tick_marks, methods_names, rotation=45)
    plt.yticks(tick_marks, methods_names)
    
    # Add numbers inside cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight='bold')
                     
    plt.ylabel('True Method', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Method', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig_cm_path = os.path.join(figures_dir, "confusion_matrix.png")
    plt.savefig(fig_cm_path, dpi=140)
    plt.close()
    print(f"Saved figure: {fig_cm_path}")
    
    # 7. Plot 2D Decision Boundary
    # Let's pick two of the most important features (say: dim, and std_range_ratio or mean_abs_d2f)
    # We find their indices in feature_names
    feat1 = 'dim'
    feat2 = 'std_range_ratio'
    
    if feat1 in feature_names and feat2 in feature_names:
        idx1 = feature_names.index(feat1)
        idx2 = feature_names.index(feat2)
        
        # Train a 2D classifier just for plotting the decision boundary
        X_2d = X_train[:, [idx1, idx2]]
        clf_2d = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf_2d.fit(X_2d, y_train)
        
        # Create grid to plot boundary
        x_min, x_max = 0.5, 5.5  # Dimension spans 1 to 5
        y_min, y_max = X[:, idx2].min() - 0.05, X[:, idx2].max() + 0.05
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.002))
        
        # Predict on mesh
        Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(9, 6.5))
        # Custom soft colors for classification regions:
        # Trapezoidal: Light Blue, Simpson: Light Green, Romberg: Light Violet, Monte Carlo: Light Coral
        cmap_light = matplotlib.colors.ListedColormap(['#D0E1F9', '#D1F2D9', '#E6D7FF', '#FFD2D2'])
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.85)
        
        # Scatter actual test samples
        scatter_colors = ['tab:blue', 'tab:green', 'tab:purple', 'tab:red']
        for class_idx, name in enumerate(methods_names):
            idx_test = np.where(y_test == class_idx)[0]
            if len(idx_test) > 0:
                plt.scatter(X_test[idx_test, idx1], X_test[idx_test, idx2],
                            color=scatter_colors[class_idx], label=name,
                            edgecolor='k', s=45, alpha=0.9)
                            
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel("Dimension (d)", fontsize=12, fontweight='bold')
        plt.ylabel("Ruggedness Ratio (std / range)", fontsize=12, fontweight='bold')
        plt.title("Integration Regime Boundary Map (ML Decision Boundary)", fontsize=13, fontweight='bold', pad=15)
        plt.legend(loc='best', framealpha=0.9, facecolor='white', edgecolor='none')
        plt.tight_layout()
        
        fig_db_path = os.path.join(figures_dir, "decision_boundary.png")
        plt.savefig(fig_db_path, dpi=140)
        plt.close()
        print(f"Saved figure: {fig_db_path}")
        
    return {
        'accuracy_dt': float(acc_dt),
        'accuracy_rf': float(acc_rf),
        'feature_names': feature_names,
        'importances': importances.tolist()
    }
