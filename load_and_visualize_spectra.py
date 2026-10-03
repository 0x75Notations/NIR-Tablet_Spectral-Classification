"""
Load and visualize the Tablets NIR spectral dataset (verified against the real file).

DATASET: NIR transmittance spectra of 310 pharmaceutical tablets
SOURCE:  http://www.models.life.ku.dk/Tablets
FILE USED: NIRdata_tablets.MAT (from the "Tablets_matlab.zip" inside the download)

CONFIRMED STRUCTURE (checked directly against the real file):
- Matrix: shape (310, 407)
    column 0 = active substance content (% w/w)
    column 1 = Tablet Type, coded 1-4 (corresponds to types A, B, C, D)
    column 2 = Production Scale, coded 0-2 (0 = laboratory, 1 = pilot, 2 = full/production)
    columns 3 onward = the actual NIR spectrum, 404 points from 7398 to 10507 cm-1
- ObjLabels: sample IDs, e.g. '01_01', '01_02', ... (310 total)
- VarLabels: column names, e.g. 'active (%w/w)', 'Type', 'Scale', then wavenumber values

HOW TO USE IN GOOGLE COLAB
---------------------------
1. Upload NIRdata_tablets.MAT into Colab's file browser (the .MAT file specifically,
   found inside Tablets.zip -> Tablets_matlab.zip -> NIRdata_tablets.MAT)
2. Update DATA_PATH below if your filename/path differs
3. Run this cell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# STEP 1: Path to the uploaded .MAT file
# ---------------------------------------------------------------------------
DATA_PATH = "NIRdata_tablets.MAT"   # <-- update this if your filename/path differs

# ---------------------------------------------------------------------------
# STEP 2: Load the .MAT file
# ---------------------------------------------------------------------------
mat = loadmat(DATA_PATH)

matrix = mat["Matrix"]          # shape (310, 407)
obj_labels = [str(x).strip() for x in mat["ObjLabels"]]   # sample IDs
var_labels = [str(x).strip() for x in mat["VarLabels"]]   # column names

print("Matrix shape:", matrix.shape)
print("Number of samples:", len(obj_labels))
print("First few variable labels:", var_labels[:6])

# ---------------------------------------------------------------------------
# STEP 3: Split into metadata columns + actual spectral columns
# ---------------------------------------------------------------------------
active_substance = matrix[:, 0]     # % w/w active ingredient
tablet_type_code = matrix[:, 1].astype(int)   # 1-4
scale_code = matrix[:, 2].astype(int)         # 0-2

spectra = matrix[:, 3:]             # shape (310, 404) - the actual NIR spectrum
wavenumbers = np.array([float(v) for v in var_labels[3:]])   # 404 wavenumber values

# Map numeric codes to human-readable labels (per the dataset's own documentation)
type_map = {1: "A", 2: "B", 3: "C", 4: "D"}
scale_map = {0: "Lab", 1: "Pilot", 2: "Full/Production"}

tablet_type = [type_map[t] for t in tablet_type_code]
scale = [scale_map[s] for s in scale_code]

# Build a tidy DataFrame for easy exploration
df = pd.DataFrame({
    "sample_id": obj_labels,
    "active_pct": active_substance,
    "tablet_type": tablet_type,
    "scale": scale,
})
print("\nSample metadata preview:")
print(df.head(10))

print("\nTablet type counts:")
print(df["tablet_type"].value_counts())

print("\nScale counts:")
print(df["scale"].value_counts())

# ---------------------------------------------------------------------------
# STEP 4: Plot example spectra, one per tablet type
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
for t in sorted(set(tablet_type)):
    idx = df.index[df["tablet_type"] == t][0]   # first example of this type
    plt.plot(wavenumbers, spectra[idx], label=f"Type {t}")

plt.xlabel("Wavenumber (cm-1)")
plt.ylabel("NIR Transmittance (absorbance units)")
plt.title("Example NIR Spectra by Tablet Type")
plt.legend()
plt.tight_layout()
plt.savefig("example_spectra_by_type.png", dpi=150)
plt.show()
print("\nSaved plot to example_spectra_by_type.png")

# ---------------------------------------------------------------------------
# STEP 5: Plot example spectra, one per production scale
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
for s in sorted(set(scale)):
    idx = df.index[df["scale"] == s][0]
    plt.plot(wavenumbers, spectra[idx], label=s)

plt.xlabel("Wavenumber (cm-1)")
plt.ylabel("NIR Transmittance (absorbance units)")
plt.title("Example NIR Spectra by Production Scale")
plt.legend()
plt.tight_layout()
plt.savefig("example_spectra_by_scale.png", dpi=150)
plt.show()
print("Saved plot to example_spectra_by_scale.png")

print("\nDone. You now have 'df' (metadata) and 'spectra' (the 310 x 404 NIR array) to build models with.")
