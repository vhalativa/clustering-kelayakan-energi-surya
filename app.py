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


def make_hist(series: pd.Series, title: str):
    fig = plt.figure()
    plt.hist(series.dropna().values, bins=30)
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)


def make_scatter(x: np.ndarray, y: np.ndarray, c: np.ndarray, title: str, xlabel: str, ylabel: str):
    fig = plt.figure()
    plt.scatter(x, y, c=c)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(fig)
    plt.close(fig)


# -----------------------------
# UI
# -----------------------------
st.sidebar.title("☀️ Solar Viability Clustering")
page = st.sidebar.radio(
    "Menu",
    ["Home", "Dataset", "Exploratory Data Analysis", "Modelling", "Clustering"],
)

# Load models once (and show a friendly error if missing)
try:
    scaler, pca, kmeans, feature_names = load_models()
except Exception as e:
    st.error(
        "Model files failed to load.\n\n"
        f"**Error:** {e}\n\n"
        "Please ensure these files exist next to app.py:\n"
        "- scaler_solar.pkl\n- pca_solar.pkl\n- kmeans_solar.pkl\n- feature_names_solar.pkl\n"
    )
    st.stop()

# Shared dataset uploader (kept in session_state)
if "uploaded_csv" not in st.session_state:
    st.session_state.uploaded_csv = None


# -----------------------------
# Pages
# -----------------------------
if page == "Home":
    st.title("Solar Viability Clustering App")
    st.write(
        """
This Streamlit app groups urban locations into clusters based on solar-related features
(e.g., sunlight hours, GHI, ROI, CO₂ reduction, etc.) using a pipeline:

**StandardScaler → PCA → KMeans**

Use the menu on the left to explore the dataset, see EDA, understand the modelling approach,
and run predictions for new inputs.
"""
    )
    
    st.subheader("Business Understanding")
    st.write(Evaluasi kelayakan energi surya)
    st.write(Tujuan dari evaluasi kelayakan energi surya adalah untuk menentukan apakah pemasangan sistem energi surya (seperti panel surya) di suatu lokasi tertentu layak secara teknis, ekonomi, dan lingkungan. Evaluasi ini membantu pengambil keputusan (individu, perusahaan, atau pemerintah) dalam merencanakan investasi energi surya dengan lebih baik. Karena dataset belum mempunyai label maka pemodelan Clustering lebih tepat untuk membuat kategori kelayakan secara otomatis Hasil cluster berfungsi untuk menginterpretasikan label secara manual, misalnya:)
    st.write(1. Cluster ke 1 → Skor viability tinggi → "Sangat Layak")
    st.write(2. Cluster ke 2 → Skor sedang → "Layak")
    st.write(2. Cluster ke 2 → Skor sedang → "Layak")
    st.write(3. Cluster ke 3 → Skor rendah → "Kurang Layak")


    
    st.subheader("Dataset Source")
    st.write("Kaggle: Urban Solar ROI and Sustainability")
    st.write("https://www.kaggle.com/datasets/shaistashahid/urban-solar-roi-and-sustainability")

    st.subheader("Required Features")
    st.code("\n".join(feature_names), language="text")

    st.info(
        "Tip: If you place the dataset file as 'solar_energy_worldwide.csv' next to app.py, "
        "the app will auto-load it. Otherwise, upload the CSV in the Dataset page."
    )

elif page == "Dataset":
    st.title("Dataset")

    st.write("Upload the Kaggle CSV file here, or place it next to `app.py` as `solar_energy_worldwide.csv`.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="dataset_uploader")
    if uploaded is not None:
        st.session_state.uploaded_csv = uploaded

    df, src_msg = get_dataset(st.session_state.uploaded_csv)
    st.caption(src_msg)

    if df is None:
        st.stop()

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Columns")
    st.write(list(df.columns))

    st.subheader("Basic Info")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", int(df.shape[0]))
    c2.metric("Columns", int(df.shape[1]))
    c3.metric("Missing values (total)", int(df.isnull().sum().sum()))

    st.subheader("Download (optional)")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download dataset as CSV", data=csv_bytes, file_name="dataset_export.csv", mime="text/csv")

elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")

    df, src_msg = get_dataset(st.session_state.uploaded_csv)
    st.caption(src_msg)

    if df is None:
        st.stop()

    df = normalize_numeric(df, feature_names + (["Solar_Viability_Score"] if "Solar_Viability_Score" in df.columns else []))

    st.subheader("Distribution (select a numeric feature)")
    feature = st.selectbox("Feature", options=[c for c in feature_names if c in df.columns], index=0)

    if feature:
        make_hist(df[feature], f"Histogram: {feature}")

    st.subheader("Correlation (features used by the model)")
    cols_present = [c for c in feature_names if c in df.columns]
    if len(cols_present) >= 2:
        corr = df[cols_present].corr(numeric_only=True)

        fig = plt.figure(figsize=(8, 6))
        plt.imshow(corr.values)
        plt.title("Correlation Heatmap (model features)")
        plt.xticks(range(len(cols_present)), cols_present, rotation=90)
        plt.yticks(range(len(cols_present)), cols_present)
        plt.colorbar()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Not enough model feature columns found in the dataset to compute correlations.")

    st.subheader("Before vs After Scaling (one sample row)")
    sample_idx = st.number_input("Row index", min_value=0, max_value=max(int(df.shape[0]-1), 0), value=0, step=1)

    if all(c in df.columns for c in feature_names):
        X_raw = df.loc[[int(sample_idx)], feature_names].copy()
        X_raw = X_raw.fillna(X_raw.mean(numeric_only=True))
        X_scaled = scaler.transform(X_raw)

        before_after = pd.DataFrame(
            {
                "Feature": feature_names,
                "Before (raw)": X_raw.iloc[0].values,
                "After (scaled)": X_scaled[0],
            }
        )
        st.dataframe(before_after, use_container_width=True)
    else:
        st.warning("Dataset does not contain all required feature columns for the before/after scaling view.")

elif page == "Modelling":
    st.title("Modelling")

    st.write(
        """
**Pipeline used**
1. **StandardScaler**: standardizes features.
2. **PCA**: reduces dimensionality.
3. **KMeans**: clusters the PCA-transformed data.

This page uses the provided pickles to show model properties and (if dataset is loaded) cluster profiles.
"""
    )

    st.subheader("Model Properties")
    c1, c2, c3 = st.columns(3)
    c1.metric("KMeans clusters", int(getattr(kmeans, "n_clusters", -1)))
    c2.metric("PCA components", int(getattr(pca, "n_components_", getattr(pca, "n_components", -1))))
    c3.metric("Features", int(len(feature_names)))

    st.subheader("PCA Explained Variance")
    if hasattr(pca, "explained_variance_ratio_"):
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)

        fig = plt.figure()
        plt.plot(range(1, len(evr) + 1), evr, marker="o")
        plt.title("Explained variance ratio per component")
        plt.xlabel("Component")
        plt.ylabel("Explained variance ratio")
        st.pyplot(fig)
        plt.close(fig)

        fig2 = plt.figure()
        plt.step(range(1, len(cum) + 1), cum, where="mid")
        plt.title("Cumulative explained variance")
        plt.xlabel("Component")
        plt.ylabel("Cumulative explained variance")
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("PCA explained variance ratio not available in this object.")

    st.subheader("Cluster Profile (requires dataset)")
    df, src_msg = get_dataset(st.session_state.uploaded_csv)
    st.caption(src_msg)

    if df is None:
        st.stop()

    profile = compute_cluster_profile(df, scaler, pca, kmeans, feature_names)

    if not profile:
        st.warning("Could not compute cluster profile. Ensure the dataset contains all required model features.")
        st.stop()

    labels = profile["labels"]
    unique, counts = np.unique(labels, return_counts=True)
    dist = pd.DataFrame({"Cluster": unique, "Count": counts}).sort_values("Cluster")
    st.dataframe(dist, use_container_width=True)

    if "viability_means" in profile:
        st.write("Mean Solar_Viability_Score per cluster:")
        means_df = pd.DataFrame(
            [{"Cluster": k, "Mean Solar_Viability_Score": v} for k, v in profile["viability_means"].items()]
        ).sort_values("Cluster")
        st.dataframe(means_df, use_container_width=True)

        st.info(
            f"Interpretation: Cluster **{profile['high_cluster']}** has the higher average viability score, "
            f"while cluster **{profile['low_cluster']}** is lower (based on your dataset)."
        )

    # Visualize in PCA space if possible (2D)
    st.subheader("Visualization in PCA Space")
    try:
        df_norm = normalize_numeric(df, feature_names)
        X = df_norm[feature_names].copy()
        if X.isnull().any().any():
            X = X.fillna(X.mean(numeric_only=True))
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        if X_pca.shape[1] >= 2:
            make_scatter(
                X_pca[:, 0],
                X_pca[:, 1],
                labels,
                "Clusters in PCA space (first 2 components)",
                "PCA 1",
                "PCA 2",
            )
        else:
            st.info("PCA has <2 components; cannot plot 2D scatter.")
    except Exception as e:
        st.warning(f"Failed to plot PCA scatter: {e}")

elif page == "Clustering":
    st.title("Clustering")

    viability_col = "Solar_Viability_Score"

    st.write(
        """
Input values for each feature and the app will output the Clustering **cluster**.

**Important note (model requirement):**
Your saved pipeline was trained using the features listed in `feature_names_solar.pkl`.
If that list includes `Solar_Viability_Score`, then the model **requires** that value at prediction time.
To fully remove data leakage, you would need to **re-train** the scaler/PCA/KMeans without that column
and export new `.pkl` files.

In this app, to keep compatibility with your current `.pkl` files:
- We show the main input fields (excluding `Solar_Viability_Score`) as the primary form.
- `Solar_Viability_Score` is placed in an *optional* expander; if you loaded the dataset, we auto-fill it
  using the dataset median as a sensible default.
"""
    )

    df, src_msg = get_dataset(st.session_state.uploaded_csv)
    st.caption(src_msg)

    # Build profile (for high/low interpretation) when dataset is available
    profile = {}
    if df is not None:
        profile = compute_cluster_profile(df, scaler, pca, kmeans, feature_names)

    # Features for UI (try to avoid leakage in the UI by hiding the viability score by default)
    ui_features = [f for f in feature_names if f != viability_col]

    # Default viability score value (used only if the trained model expects it)
    default_viability = 0.0
    if df is not None and viability_col in df.columns:
        try:
            default_viability = float(pd.to_numeric(df[viability_col], errors="coerce").median())
        except Exception:
            default_viability = 0.0

    st.subheader("Input Features")
    with st.form("predict_form"):
        cols = st.columns(3)
        inputs = {}

        # Main inputs (without viability score)
        for i, feat in enumerate(ui_features):
            with cols[i % 3]:
                inputs[feat] = st.number_input(feat, value=0.0, format="%.6f")

        # Optional viability score (only used if the trained model expects it)
        viability_input = None
        if viability_col in feature_names:
            with st.expander("Optional (trained feature): Solar_Viability_Score", expanded=False):
                st.caption(
                    "Your saved model was trained with this column. "
                    "If you don't know the value, you may keep the default (median from dataset if loaded)."
                )
                viability_input = st.number_input(
                    viability_col,
                    value=float(default_viability),
                    format="%.6f",
                    help="Required by the current saved pipeline. Remove only after retraining the model without it.",
                )

        submitted = st.form_submit_button("Clustering Cluster")

    if submitted:
        # Ensure we pass the exact feature order expected by the trained pipeline
        full_inputs = {}
        for feat in feature_names:
            if feat == viability_col:
                if viability_input is None:
                    # Should not happen if viability_col is in feature_names, but keep safe fallback
                    full_inputs[feat] = float(default_viability)
                else:
                    full_inputs[feat] = float(viability_input)
            else:
                full_inputs[feat] = float(inputs.get(feat, 0.0))

        X_new = pd.DataFrame([full_inputs], columns=feature_names)
        X_scaled = scaler.transform(X_new)
        X_pca = pca.transform(X_scaled)
        cluster = int(kmeans.predict(X_pca)[0])

        st.success(f"Clustering Cluster: {cluster}")

        if profile and "high_cluster" in profile:
            if cluster == profile["high_cluster"]:
                st.info("Interpretation: **Higher viability** cluster (based on your loaded dataset).")
            elif cluster == profile["low_cluster"]:
                st.info("Interpretation: **Lower viability** cluster (based on your loaded dataset).")
            else:
                st.info("Interpretation: Cluster label found, but viability profile is unclear.")
        else:
            st.caption("Load the dataset to enable higher/lower viability interpretation.")

        st.subheader("Pipeline Output (debug view)")
        st.write("Scaled features:")
        st.dataframe(pd.DataFrame([X_scaled[0]], columns=feature_names), use_container_width=True)
        st.write("PCA vector:")
        st.dataframe(pd.DataFrame([X_pca[0]], columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]), use_container_width=True)
