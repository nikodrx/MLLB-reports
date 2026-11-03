# PRDL & MLLB 2026-27 — Machine Learning and Deep Learning projects

Code for my PRDL (Predictive and Descriptive Learning) and MLLB (Machine Learning Lab) reports, UPM, 2026-27.

| Part | Case study | Folder |
|---|---|---|
| Machine Learning | Obstructive Sleep Apnea (OSA) from clinical data: regression (IAH), classification (Healthy vs Severe), clustering | [`machine-learning/`](machine-learning/) |
| Deep Learning | To be defined | [`deep-learning/`](deep-learning/) |

## Repository structure

```
machine-learning/
├── data/raw/          original Excel file (not uploaded, see below)
├── data/processed/    cleaned datasets produced by 01_data_preparation
├── src/               Python scripts
├── notebooks/         Jupyter notebooks (EDA, regression, classification, clustering)
├── figures/           plots used in the reports
└── results/           result tables (metrics per model)
deep-learning/         same structure for the DL project
reports/               PDF versions of the submitted reports
```

## How to run

```bash
pip install -r requirements.txt
python machine-learning/src/01_data_preparation.py machine-learning/data/raw/Info_BDApnea_QuironMalaga.xlsx
```

The clinical database (`Info_BDApnea_QuironMalaga.xlsx`) is provided on the course Moodle and is not included in this repository because it contains patient data. Place it in `machine-learning/data/raw/` before running the code.

## Machine Learning pipeline

| Step | File | Input | Output |
|---|---|---|---|
| 1. Data preparation (ETL) | `src/01_data_preparation.py` | raw Excel file | `OSA_DB_UPM.xlsx`, `OSA_DB_UPM_BMI_classes.xlsx`, `OSA_extreme_male.xlsx` |
| 2. Exploratory Data Analysis | `notebooks/02_eda.ipynb` | `OSA_DB_UPM_BMI_classes.xlsx` | figures |
| 3. Regression (IAH) | `notebooks/03_regression.ipynb` | `OSA_DB_UPM.xlsx` | metrics, figures |
| 4. Classification (Healthy vs Severe) | `notebooks/04_classification.ipynb` | `OSA_extreme_male.xlsx` | metrics, figures |
| 5. Clustering | `notebooks/05_clustering.ipynb` | `OSA_DB_UPM_BMI_classes.xlsx` | figures |

## Use of AI assistants

The use of AI code assistants (GitHub Copilot, Gemini, Claude) and the AI Error Bounty cases are documented in the MLLB reports.
