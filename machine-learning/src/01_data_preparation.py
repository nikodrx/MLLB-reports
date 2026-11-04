"""
OSA case study - Data preparation (ETL) for the PRDL / MLLB ML project.

Follows the steps of the course slides "MLLB_1_OSA_CaseStudy_DataPreparation":
  Info_BDApnea_QuironMalaga.xlsx  ->  OSA_DB_UPM.xlsx  (clinical variables only)
  then: classes Healthy (IAH <= 10) / Severe (IAH >= 30), BMI (feature engineering),
  and a male-only subset of extreme cases for classification.

Usage:
    python osa_data_preparation.py                          # file in the current folder
    python osa_data_preparation.py path/to/Info_BDApnea_QuironMalaga.xlsx

The "# %%" markers let you run it cell by cell in VS Code or Jupyter.
Requires: pandas, openpyxl  (pip install pandas openpyxl)
"""

# %% 0. Imports and paths
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]   # .../machine-learning
RAW_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else ML_DIR / "data" / "raw" / "Info_BDApnea_QuironMalaga.xlsx"
OUT_DIR = ML_DIR / "data" / "processed"

# Thresholds used in class (slide 19)
HEALTHY_MAX_IAH = 10   # IAH <= 10  -> Healthy
SEVERE_MIN_IAH = 30    # IAH >= 30  -> Severe

# %% 1. EXTRACT: read the raw Excel file
raw = pd.read_excel(RAW_FILE, sheet_name="Hoja1")
print(f"Raw file: {raw.shape[0]} rows x {raw.shape[1]} columns")

# %% 2. Keep only real patient rows
# The sheet contains empty rows and one free-text note at the bottom.
# Real patients have an ID such as "P0001".
is_patient = raw["Patient"].astype(str).str.fullmatch(r"P\d{4}")
df = raw.loc[is_patient].copy()
print(f"Patient rows: {len(df)}  (removed {len(raw) - len(df)} empty / non-patient rows)")

# P0363 appears twice with different data (two different patients, ID typo):
# keep both rows and make the second ID unique.
dup = df["Patient"].duplicated(keep="first")
if dup.any():
    print("Duplicated IDs (different patients):", df.loc[dup, "Patient"].tolist())
    df.loc[dup, "Patient"] = df.loc[dup, "Patient"] + "_b"

# %% 3. Select the clinical variables and give them English names
columns = {
    "Patient": "Patient",
    "Gender": "Gender",
    "IAH": "IAH",               # Apnea-Hypopnea Index (target)
    "Peso": "Weight",           # kg
    "Talla": "Height",          # cm
    "Edad": "Age",              # years
    "PerCervical": "Cervical",  # neck circumference, cm
}
df = df[list(columns)].rename(columns=columns)

# %% 4. TRANSFORM: fix data types and missing-value codes
num_cols = ["IAH", "Weight", "Height", "Age", "Cervical"]

# Weight is stored as text in some rows ("no se saben", "No quiere saberlo")
# -> to_numeric turns any non-numeric text into NaN.
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# In this database "-1" means "value not available" -> replace by NaN
df[num_cols] = df[num_cols].replace(-1, np.nan)

df["Gender"] = df["Gender"].str.strip().str.lower()   # "hombre" / "mujer"

print("\nMissing values per column (after recoding -1 and text as NaN):")
print(df.isna().sum().to_string())

# %% 5. Handle missing values: keep complete cases only
# (simple and transparent; imputation can be compared later inside the ML pipeline)
n_before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"\nComplete cases: {len(df)} patients (removed {n_before - len(df)} with missing values)")

# %% 6. Sanity checks on physiological ranges
ranges = {"IAH": (0, 150), "Weight": (30, 250), "Height": (120, 220),
          "Age": (15, 100), "Cervical": (25, 70)}
for c, (lo, hi) in ranges.items():
    bad = df[(df[c] < lo) | (df[c] > hi)]
    if len(bad):
        print(f"WARNING {c}: {len(bad)} values outside [{lo}, {hi}]\n", bad)
print("\nSummary statistics:")
print(df[num_cols].describe().round(1).to_string())
print("\nGender:", df["Gender"].value_counts().to_dict())

# %% 7. LOAD: save the clean database used in class
df.to_excel(OUT_DIR / "OSA_DB_UPM.xlsx", index=False)
print(f"\nSaved {OUT_DIR / 'OSA_DB_UPM.xlsx'}")

# %% 8. Feature engineering: BMI = weight (kg) / height (m)^2
df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2

# %% 9. Classification dataset: Healthy (IAH <= 10) vs Severe (IAH >= 30)
# Intermediate patients (10 < IAH < 30) are left out.
df["OSA"] = np.select(
    [df["IAH"] <= HEALTHY_MAX_IAH, df["IAH"] >= SEVERE_MIN_IAH],
    ["Healthy", "Severe"],
    default="Intermediate",
)
print("\nOSA classes (all patients):", df["OSA"].value_counts().to_dict())

extreme = df[df["OSA"] != "Intermediate"].reset_index(drop=True)
extreme_male = extreme[extreme["Gender"] == "hombre"].reset_index(drop=True)
print("Extreme cases:", extreme["OSA"].value_counts().to_dict())
print("Extreme cases, male only:", extreme_male["OSA"].value_counts().to_dict())

df.to_excel(OUT_DIR / "OSA_DB_UPM_BMI_classes.xlsx", index=False)
extreme_male.to_excel(OUT_DIR / "OSA_extreme_male.xlsx", index=False)
print(f"Saved {OUT_DIR / 'OSA_DB_UPM_BMI_classes.xlsx'} and {OUT_DIR / 'OSA_extreme_male.xlsx'}")
