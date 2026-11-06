# Deep Learning project — Radar target classification (Real Doppler RAD-DAR)

Classification of radar targets (**Cars, Drones, People**) from range-Doppler maps (11 × 61) measured with an FMCW radar, comparing deep learning architectures: Dense, CNN, RNN (LSTM/GRU) and Transformer, with ablation studies.

## Dataset

**Real Doppler RAD-DAR database** (Kaggle): https://www.kaggle.com/datasets/iroldan/real-doppler-raddar-database

The data is not included in this repository. To run the notebooks:

1. Download the dataset from Kaggle ("Download dataset as zip").
2. Extract it into `deep-learning/data/raw/`.

## Structure

```
deep-learning/
├── data/raw/      dataset from Kaggle (not uploaded)
├── notebooks/     one notebook per step (data loading, EDA, models, ablations)
├── src/           shared Python code (data loading, training utilities)
├── figures/       plots used in the reports
└── results/       result tables
```
