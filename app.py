# app.py
# Streamlit app: Clustering Evaluasi Kelayakan Energi Surya
#
# Pages:
# - Home
# - Dataset
# - Exploratory Data Analysis (EDA)
# - Modelling
# - Clustering
#
# Notes:
# - This app is configured to work with the provided .pkl files (scaler, pca, kmeans, feature names).
# - Kaggle datasets usually require authentication for programmatic download. This app therefore supports:
#   (1) Uploading the CSV manually, or
#   (2) Placing the CSV in the project folder (default name: solar_energy_worldwide.csv)

import io
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

# NEW (for elbow & silhouette like in the .ipynb)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="Solar Viability Clustering",
    page_icon="☀️",
    layout="wide",
)


# -----------------------------
# Pickle compatibility loader
# (fix for pickles referencing numpy._core.*)
# -----------------------------
class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

def compat_load(path: str):
    with open(path, "rb") as f:
        return CompatUnpickler(f).load()


# -----------------------------
# Paths (edit if needed)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET_FILENAME = "solar_energy_worldwide.csv"  # from the notebook
DEFAULT_DATASET_PATH = BASE_DIR / DEFAULT_DATASET_FILENAME

SCALER_PATH = BASE_DIR / "scaler_solar.pkl"
PCA_PATH = BASE_DIR / "pca_solar.pkl"
KMEANS_PATH = BASE_DIR / "kmeans_solar.pkl"
FEATURES_PATH = BASE_DIR / "feature_names_solar.pkl"


# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_models() -> Tuple[object, object, object, list]:
    missing = [p.name for p in [SCALER_PATH, PCA_PATH, KMEANS_PATH, FEATURES_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files: "
            + ", ".join(missing)
            + "\n\nPlace them next to app.py, or update the paths inside app.py."
        )

    scaler = compat_load(str(SCALER_PATH))
    pca = compat_load(str(PCA_PATH))
    kmeans = compat_load(str(KMEANS_PATH))
    feature_names = compat_load(str(FEATURES_PATH))

    if not isinstance(feature_names, list) or len(feature_names) == 0:
        raise ValueError("feature_names_solar.pkl must contain a non-empty list of feature names.")

    return scaler, pca, kmeans, feature_names


# -----------------------------
# Dataset loading helpers
# -----------------------------
@st.cache_data
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data
def read_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def normalize_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return df2

def get_dataset(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Returns (df, source_message)
    """
    if uploaded_file is not None:
        df = read_csv_bytes(uploaded_file.getvalue())
        return df, f"Loaded from upload: {uploaded_file.name}"

    if DEFAULT_DATASET_PATH.exists():
        df = read_csv_path(str(DEFAULT_DATASET_PATH))
        return df, f"Loaded from local file: {DEFAULT_DATASET_PATH.name}"

    return None, (
        "Dataset not loaded yet. Upload a CSV on this page, or place "
        f"'{DEFAULT_DATASET_FILENAME}' next to app.py."
    )


# -----------------------------
# Cluster interpretation helpers
# -----------------------------
def compute_cluster_profile(
    df: pd.DataFrame,
    scaler,
    pca,
    kmeans,
    feature_names: list,
    viability_col: str = "Solar_Viability_Score",
) -> Dict:
    """
    Compute cluster means (esp. viability score) to label clusters as 'Higher/Lower viability'.
    If required columns are missing, return empty profile.
    """
    if df is None:
        return {}

    df = normalize_numeric(df, feature_names + ([viability_col] if viability_col else []))
    if not all(c in df.columns for c in feature_names):
        return {}

    X = df[feature_names].copy()
    if X.isnull().any().any():
        X = X.fillna(X.mean(numeric_only=True))

    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    labels = kmeans.predict(X_pca)

    out = {"labels": labels}

    if viability_col in df.columns:
        tmp = df.copy()
        tmp["Cluster"] = labels
        means = tmp.groupby("Cluster")[viability_col].mean().sort_values()
        out["viability_means"] = means.to_dict()

        # identify highest viability cluster
        out["low_cluster"] = int(means.index[0])
        out["high_cluster"] = int(means.index[-1])

    return out


# -----------------------------
# Plot helpers (matplotlib)
# -----------------------------
def make_hist(series: pd.Series, title: str, bins: int = 30):
    fig = plt.figure()
    plt.hist(series.dropna().values, bins=bins)
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

def make_scatter(x: np.ndarray, y: np.ndarray, c: Optional[np.ndarray], title: str, xlabel: str, ylabel: str):
    fig = plt.figure()
    if c is None:
        plt.scatter(x, y)
    else:
        plt.scatter(x, y, c=c)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(fig)
    plt.close(fig)

def make_line(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(x, y, marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    st.pyplot(fig)
    plt.close(fig)

def make_bar(categories: list, values: list, title: str, xlabel: str, ylabel: str, rotate_xticks: int = 45):
    fig = plt.figure()
    plt.bar(categories, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate_xticks)
    st.pyplot(fig)
    plt.close(fig)

def make_corr_heatmap(df: pd.DataFrame, cols: list, title: str = "Correlation Heatmap (model features)"):
    """
    Notebook kamu pakai seaborn heatmap (annot=True).
    Di sini kita buat versi matplotlib-only, tetap ada angka korelasi (annotasi).
    """
    corr = df[cols].corr(numeric_only=True)
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(corr.values)
    plt.title(title)
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.colorbar()

    # annotate values like seaborn annot=True
    for i in range(le)
