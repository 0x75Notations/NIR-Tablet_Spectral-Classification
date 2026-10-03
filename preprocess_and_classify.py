"""
Preprocess NIR spectra and build classification models to predict Tablet Type.

BUILDS ON: load_and_visualize_spectra.py (same dataset, same file)
GOAL: classify tablets into Type A/B/C/D based on their NIR spectrum.

WHAT THIS SCRIPT DOES
----------------------
1. Reloads the data (same as before)
2. Preprocesses spectra: Savitzky-Golay smoothing + SNV normalization
   (this is the "chemistry-aware" step - removes noise/baseline shifts
   that have nothing to do with the actual chemical composition)
3. Splits into train/test sets
4. Trains 3 models: Logistic Regression, Random Forest, XGBoost
5. Evaluates each: accuracy, confusion matrix, classification report
6. Shows feature importance - WHICH wavelengths mattered most
   (this is what Fatima will interpret chemically)

HOW TO USE IN COLAB
--------------------
1. Make sure NIRdata_tablets.MAT is already uploaded (same file as before)
2. Paste this entire script into a NEW code cell (keep it separate from
   the first script, or combine them - either works)
3. Run the cell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import savgol_filter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# STEP 1: Reload the data (same as the first script)
# ---------------------------------------------------------------------------
DATA_PATH = "NIRdata_tablets.MAT"   # <-- update to match your uploaded filename

mat = loadmat(DATA_PATH)
matrix = mat["Matrix"]
var_labels = [str(x).strip() for x in mat["VarLabels"]]

active_substance = matrix[:, 0]
tablet_type_code = matrix[:, 1].astype(int)
scale_code = matrix[:, 2].astype(int)
spectra_raw = matrix[:, 3:]
wavenumbers = np.array([float(v) for v in var_labels[3:]])

type_map = {1: "A", 2: "B", 3: "C", 4: "D"}
tablet_type = np.array([type_map[t] for t in tablet_type_code])

print("Loaded", spectra_raw.shape[0], "spectra with", spectra_raw.shape[1], "wavelength points each")

# ---------------------------------------------------------------------------
# STEP 2: Preprocess - Savitzky-Golay smoothing + SNV normalization
# ---------------------------------------------------------------------------
# Savitzky-Golay: smooths out random noise while preserving peak shapes
spectra_smoothed = savgol_filter(spectra_raw, window_length=11, polyorder=2, axis=1)

# SNV (Standard Normal Variate): removes scaling/baseline differences between
# samples so the model focuses on spectral SHAPE, not just overall brightness
def snv(spectra):
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    return (spectra - mean) / std

spectra_processed = snv(spectra_smoothed)

# Visual check: compare one raw vs. processed spectrum
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(wavenumbers, spectra_raw[0])
plt.title("Raw Spectrum (sample 0)")
plt.xlabel("Wavenumber (cm-1)")

plt.subplot(1, 2, 2)
plt.plot(wavenumbers, spectra_processed[0])
plt.title("After Smoothing + SNV (sample 0)")
plt.xlabel("Wavenumber (cm-1)")
plt.tight_layout()
plt.savefig("preprocessing_comparison.png", dpi=150)
plt.show()
print("Saved plot to preprocessing_comparison.png")

# ---------------------------------------------------------------------------
# STEP 3: Train/test split
# ---------------------------------------------------------------------------
le = LabelEncoder()
y = le.fit_transform(tablet_type)   # converts A/B/C/D into 0/1/2/3
X = spectra_processed

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# STEP 4: Train and evaluate 3 models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "accuracy": acc, "y_pred": y_pred}

    print(f"\n{'='*50}")
    print(f"{name}  |  Accuracy: {acc:.3f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# ---------------------------------------------------------------------------
# STEP 5: Confusion matrices for all 3 models
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name}\nAcc: {res['accuracy']:.3f}")
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_)
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
print("\nSaved plot to confusion_matrices.png")

# ---------------------------------------------------------------------------
# STEP 6: Feature importance - which wavelengths mattered most
# (This is what Fatima interprets chemically)
# ---------------------------------------------------------------------------
rf_importances = results["Random Forest"]["model"].feature_importances_

plt.figure(figsize=(10, 5))
plt.plot(wavenumbers, rf_importances)
plt.xlabel("Wavenumber (cm-1)")
plt.ylabel("Importance (Random Forest)")
plt.title("Which Wavelengths Matter Most for Classifying Tablet Type")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("Saved plot to feature_importance.png")

top_indices = np.argsort(rf_importances)[-10:][::-1]
print("\nTop 10 most important wavenumbers (for Fatima to interpret chemically):")
for idx in top_indices:
    print(f"  {wavenumbers[idx]:.1f} cm-1  (importance: {rf_importances[idx]:.4f})")

print("\nDone. Models trained, evaluated, and top wavelengths identified for chemical interpretation.")
