"""
Shared utilities for the deep learning notebooks (Real Doppler RAD-DAR).

- load_data():          maps, labels and the fixed session-based train / val / test split
- normalise():          scaling with statistics of the training set only
- crop_center():        keep only the central Doppler bins (removes the vertical stripes)
- physical_features():  3 hand-crafted features from the EDA
- evaluate():           accuracy, per-class recall and confusion matrix
- plot_history():       learning curves of a Keras model
- plot_confusion():     confusion matrix figure
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

DL_DIR = Path(__file__).resolve().parents[1]          # .../deep-learning
DATA_FILE = DL_DIR / "data" / "processed" / "raddar.npz"

CENTER_DOPPLER = slice(24, 37)   # central Doppler bins around the target (bins 24..36)


def load_data(path=DATA_FILE):
    """Return a dict with X (N, 11, 61), y, class_names, sessions and the split indices."""
    d = np.load(path, allow_pickle=True)
    return {
        "X": d["X"].astype(np.float32),
        "y": d["y"].astype(np.int64),
        "class_names": [str(c) for c in d["class_names"]],
        "sessions": d["sessions"],
        "train": d["train_idx"], "val": d["val_idx"], "test": d["test_idx"],
    }


def crop_center(X):
    """Keep only the central Doppler bins (11 x 13 maps). Removes stripes far from the target."""
    return X[:, :, CENTER_DOPPLER]


def normalise(X, train_idx):
    """Standardise with ONE mean and ONE std computed on the training maps only.
    A single scalar keeps the relative structure of each map (no per-pixel distortion)."""
    mu, sd = X[train_idx].mean(), X[train_idx].std()
    return (X - mu) / sd


def physical_features(X):
    """Peak power above noise (dB), Doppler width and range extent (bins > 10 dB above noise)."""
    n = len(X)
    noise = np.median(X.reshape(n, -1), axis=1)
    center = X[:, 2:9, 20:41]
    peak = np.array([np.unravel_index(m.argmax(), m.shape) for m in center])
    r_pk, d_pk = peak[:, 0] + 2, peak[:, 1] + 20
    rel = X - noise[:, None, None]
    return pd.DataFrame({
        "peak_db": center.reshape(n, -1).max(1) - noise,
        "doppler_width": (rel[np.arange(n), r_pk, :] > 10).sum(1),
        "range_extent": (rel[np.arange(n), :, d_pk] > 10).sum(1),
    })


def evaluate(y_true, y_pred, class_names):
    """Accuracy and recall of each class."""
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    rec = recall_score(y_true, y_pred, average=None, labels=range(len(class_names)))
    for c, r in zip(class_names, rec):
        out[f"recall {c}"] = r
    return out


def plot_history(history, title, ax_loss=None, ax_acc=None):
    """Training vs validation loss and accuracy per epoch."""
    h = history.history if hasattr(history, "history") else history
    if ax_loss is None:
        _, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    ep = np.arange(1, len(h["loss"]) + 1)
    ax_loss.plot(ep, h["loss"], label="train"); ax_loss.plot(ep, h["val_loss"], label="validation")
    ax_acc.plot(ep, h["accuracy"], label="train"); ax_acc.plot(ep, h["val_accuracy"], label="validation")
    best = int(np.argmin(h["val_loss"])) + 1
    for ax in (ax_loss, ax_acc):
        ax.axvline(best, color="grey", ls=":", lw=1)
        ax.set_xlabel("Epoch"); ax.legend()
    ax_loss.set(title=f"{title}: loss", ylabel="Cross-entropy loss")
    ax_acc.set(title=f"{title}: accuracy", ylabel="Accuracy")


def plot_confusion(y_true, y_pred, class_names, title, ax):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title); ax.grid(False)
