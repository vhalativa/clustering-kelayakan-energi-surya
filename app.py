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
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    st.pyplot(fig)
    plt.close(fig)

def make_elbow_silhouette(X_pca: np.ndarray, k_min: int = 2, k_max: int = 10):
    """
    Replikasi grafik Elbow + Silhouette dari notebook:
    - inertia vs jumlah cluster
    - silhouette score vs jumlah cluster
    """
    K = list(range(k_min, k_max))
    inertia = []
    sil_scores = []

    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_pca)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(X_pca, km.labels_))

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(K, inertia, marker="o")
    ax1.set_xlabel("Jumlah Cluster")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")

    ax2.plot(K, sil_scores, marker="o")
    ax2.set_xlabel("Jumlah Cluster")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Scores")

    st.pyplot(fig)
    plt.close(fig)

    best_k = K[int(np.argmax(sil_scores))]
    return {"K": K, "inertia": inertia, "silhouette": sil_scores, "best_k": best_k}


# -----------------------------
# UI
# -----------------------------
st.sidebar.title("☀️ Solar Viability Clustering")
page = st.sidebar.radio(
    "Menu",
    ["Home", "Dataset", "Exploratory Data Analysis", "Modelling", "Clustering", "About"],
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
    st.write("""
Evaluasi kelayakan energi surya

Tujuan dari evaluasi kelayakan energi surya adalah untuk menentukan apakah pemasangan sistem energi surya (seperti panel surya) di suatu lokasi tertentu layak secara teknis, ekonomi, dan lingkungan. Evaluasi ini membantu pengambil keputusan (individu, perusahaan, atau pemerintah) dalam merencanakan investasi energi surya dengan lebih baik.

Karena dataset belum mempunyai label, maka pemodelan **Clustering** lebih tepat untuk membuat kategori kelayakan secara otomatis.  
Hasil cluster berfungsi untuk menginterpretasikan label secara manual, misalnya:

1. Cluster ke-1 → Skor viability tinggi → **"Sangat Layak"**
2. Cluster ke-2 → Skor sedang → **"Layak"**
3. Cluster ke-3 → Skor rendah → **"Kurang Layak"**
""")

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

    st.subheader("Grafik dari Notebook (.ipynb) — EDA Utama")

    # 1) Bar: Jumlah kota per Region
    with st.expander("1) Bar Chart — Jumlah Kota per Region", expanded=True):
        st.write(
            "Grafik ini menunjukkan **distribusi jumlah kota** pada setiap *Region*. "
            "Tujuannya untuk melihat apakah dataset seimbang antar wilayah atau didominasi region tertentu."
        )
        if "Region" in df.columns:
            region_counts = df["Region"].value_counts()
            make_bar(
                categories=list(region_counts.index),
                values=list(region_counts.values),
                title="Jumlah Kota per Region",
                xlabel="Region",
                ylabel="Jumlah Kota",
                rotate_xticks=45,
            )
        else:
            st.warning("Kolom `Region` tidak ditemukan di dataset.")

    # 2) Histogram: Solar Viability Score
    with st.expander("2) Histogram — Distribusi Solar_Viability_Score", expanded=False):
        st.write(
            "Histogram ini memperlihatkan **sebaran skor kelayakan** (Solar_Viability_Score). "
            "Kamu bisa melihat apakah skor cenderung banyak di nilai rendah/tinggi, serta indikasi outlier."
        )
        if "Solar_Viability_Score" in df.columns:
            make_hist(df["Solar_Viability_Score"], "Distribusi Solar Viability Score", bins=30)
        else:
            st.warning("Kolom `Solar_Viability_Score` tidak ditemukan di dataset.")

    # 3) Line: Latitude vs Viability (sorted)
    with st.expander("3) Line Chart — Solar Viability Score across Cities by Latitude", expanded=False):
        st.write(
            "Grafik garis ini mengurutkan data berdasarkan **Latitude**, lalu memplot **Solar_Viability_Score**. "
            "Tujuannya untuk melihat **pola tren** kelayakan terhadap posisi lintang secara lebih halus."
        )
        if ("Latitude" in df.columns) and ("Solar_Viability_Score" in df.columns):
            df_sorted_by_latitude = df.sort_values(by="Latitude").reset_index(drop=True)
            make_line(
                x=df_sorted_by_latitude["Latitude"].to_numpy(),
                y=df_sorted_by_latitude["Solar_Viability_Score"].to_numpy(),
                title="Solar Viability Score across Cities by Latitude",
                xlabel="Latitude",
                ylabel="Solar Viability Score",
            )
        else:
            st.warning("Butuh kolom `Latitude` dan `Solar_Viability_Score`.")

    # 4) Scatter: Latitude vs Viability
    with st.expander("4) Scatter — Latitude vs Solar Viability Score", expanded=False):
        st.write(
            "Scatter plot ini menampilkan **hubungan langsung** antara `Latitude` dan `Solar_Viability_Score`. "
            "Kalau titik-titik membentuk pola tertentu, berarti ada korelasi/relasi yang menarik untuk dianalisis."
        )
        if ("Latitude" in df.columns) and ("Solar_Viability_Score" in df.columns):
            make_scatter(
                x=df["Latitude"].to_numpy(),
                y=df["Solar_Viability_Score"].to_numpy(),
                c=None,
                title="Latitude vs Solar Viability Score",
                xlabel="Latitude",
                ylabel="Solar Viability Score",
            )
        else:
            st.warning("Butuh kolom `Latitude` dan `Solar_Viability_Score`.")

    # 5) Scatter: Estimated Annual Savings USD vs Viability
    with st.expander("5) Scatter — Estimated Annual Savings USD vs Solar Viability Score", expanded=False):
        st.write(
            "Scatter plot ini mengecek apakah **penghematan tahunan (USD)** cenderung lebih tinggi "
            "pada lokasi dengan **skor kelayakan** yang lebih tinggi."
        )
        if ("Estimated_Annual_Savings_USD" in df.columns) and ("Solar_Viability_Score" in df.columns):
            make_scatter(
                x=df["Estimated_Annual_Savings_USD"].to_numpy(),
                y=df["Solar_Viability_Score"].to_numpy(),
                c=None,
                title="Estimated Annual Savings USD vs Solar Viability Score",
                xlabel="Estimated Annual Savings USD",
                ylabel="Solar Viability Score",
            )
        else:
            st.warning("Butuh kolom `Estimated_Annual_Savings_USD` dan `Solar_Viability_Score`.")

    # 6) Scatter: CO2 Reduction vs Viability
    with st.expander("6) Scatter — CO2 Reduction Tons per Year vs Solar Viability Score", expanded=False):
        st.write(
            "Scatter plot ini menunjukkan apakah lokasi dengan **potensi reduksi CO₂** lebih besar "
            "juga memiliki **skor kelayakan** lebih tinggi."
        )
        if ("CO2_Reduction_Tons_per_Year" in df.columns) and ("Solar_Viability_Score" in df.columns):
            make_scatter(
                x=df["CO2_Reduction_Tons_per_Year"].to_numpy(),
                y=df["Solar_Viability_Score"].to_numpy(),
                c=None,
                title="CO2 Reduction Tons per Year vs Solar Viability Score",
                xlabel="CO2 Reduction Tons per Year",
                ylabel="Solar Viability Score",
            )
        else:
            st.warning("Butuh kolom `CO2_Reduction_Tons_per_Year` dan `Solar_Viability_Score`.")

    # 7) Scatter: ROI Percentage vs Viability
    with st.expander("7) Scatter — ROI Percentage vs Solar Viability Score", expanded=False):
        st.write(
            "Scatter plot ini menilai apakah **ROI (%)** cenderung meningkat ketika "
            "**Solar_Viability_Score** meningkat. Ini penting untuk argumen kelayakan ekonomi."
        )
        if ("ROI_Percentage" in df.columns) and ("Solar_Viability_Score" in df.columns):
            make_scatter(
                x=df["ROI_Percentage"].to_numpy(),
                y=df["Solar_Viability_Score"].to_numpy(),
                c=None,
                title="ROI Percentage vs Solar Viability Score",
                xlabel="ROI Percentage",
                ylabel="Solar Viability Score",
            )
        else:
            st.warning("Butuh kolom `ROI_Percentage` dan `Solar_Viability_Score`.")

    st.divider()

    st.subheader("Distribution (select a numeric feature)")
    feature = st.selectbox("Feature", options=[c for c in feature_names if c in df.columns], index=0)
    if feature:
        st.write(
            "Grafik ini membantu melihat **sebaran** salah satu fitur yang dipakai model. "
            "Kalau distribusinya sangat skew/outlier berat, scaling dan PCA jadi semakin penting."
        )
        make_hist(df[feature], f"Histogram: {feature}")

    st.subheader("Correlation (features used by the model)")
    cols_present = [c for c in feature_names if c in df.columns]
    if len(cols_present) >= 2:
        st.write(
            "Heatmap korelasi ini mengecek **hubungan linear antar fitur**. "
            "Korelasi tinggi bisa berarti ada fitur yang redundan; PCA membantu merangkum informasi tersebut."
        )
        make_corr_heatmap(df, cols_present, title="Correlation Heatmap (model features)")
    else:
        st.warning("Not enough model feature columns found in the dataset to compute correlations.")

    st.subheader("Before vs After Scaling (one sample row)")
    st.write(
        "Tabel ini menunjukkan **nilai sebelum scaling** dan **sesudah scaling** (StandardScaler) untuk satu baris data. "
        "Scaling memastikan fitur dengan skala besar tidak mendominasi PCA/KMeans."
    )
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

Halaman ini menambahkan grafik-grafik dari notebook (.ipynb) yang menjelaskan PCA dan pemilihan jumlah cluster.
"""
    )

    st.subheader("Model Properties")
    c1, c2, c3 = st.columns(3)
    c1.metric("KMeans clusters (saved model)", int(getattr(kmeans, "n_clusters", -1)))
    c2.metric("PCA components (saved model)", int(getattr(pca, "n_components_", getattr(pca, "n_components", -1))))
    c3.metric("Features", int(len(feature_names)))

    st.subheader("PCA Explained Variance (grafik notebook: bar + cumulative step)")
    st.write(
        "Grafik ini menunjukkan kontribusi varians tiap komponen PCA (**bar**) dan akumulasi varians (**step**). "
        "Tujuannya untuk menilai apakah jumlah komponen PCA sudah cukup mewakili informasi data."
    )

    if hasattr(pca, "explained_variance_ratio_"):
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)

        fig = plt.figure(figsize=(6, 4))
        plt.bar(range(len(evr)), evr, alpha=0.7, label="Individual explained variance")
        plt.step(range(len(cum)), cum, where="mid", label="Cumulative explained variance")
        plt.ylabel("Explained variance ratio")
        plt.xlabel("Principal components")
        plt.legend(loc="best")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Scree Plot (grafik notebook)")
        st.write(
            "Scree plot memvisualisasikan **varians per komponen**. "
            "Jika mulai datar setelah komponen tertentu, itu tanda tambahan komponen berikutnya memberi manfaat kecil."
        )
        fig2 = plt.figure()
        plt.plot(range(1, len(evr) + 1), evr, marker="o")
        plt.xlabel("Komponen Utama ke-")
        plt.ylabel("Varians (Nilai Eigen relatif)")
        plt.title("Scree Plot")
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
        st.write(
            "Tabel ini menampilkan **rata-rata Solar_Viability_Score** per cluster "
            "untuk membantu interpretasi cluster (mis. cluster paling tinggi = paling layak)."
        )
        means_df = pd.DataFrame(
            [{"Cluster": k, "Mean Solar_Viability_Score": v} for k, v in profile["viability_means"].items()]
        ).sort_values("Cluster")
        st.dataframe(means_df, use_container_width=True)

        st.info(
            f"Interpretation: Cluster **{profile['high_cluster']}** has the higher average viability score, "
            f"while cluster **{profile['low_cluster']}** is lower (based on your dataset)."
        )

    st.subheader("Elbow & Silhouette (grafik notebook) — cari K optimal dari dataset")
    st.write(
        "Grafik ini mereplikasi notebook: "
        "**Elbow Method** melihat penurunan inertia (semakin kecil semakin rapat), "
        "dan **Silhouette Score** menilai kualitas pemisahan cluster (semakin besar semakin baik). "
        "Ini dihitung ulang dari dataset yang kamu load (bukan dari model KMeans yang sudah tersimpan)."
    )

    try:
        df_norm = normalize_numeric(df, feature_names)
        X = df_norm[feature_names].copy()
        if X.isnull().any().any():
            X = X.fillna(X.mean(numeric_only=True))
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

        if X_pca.shape[0] >= 10:
            res = make_elbow_silhouette(X_pca, k_min=2, k_max=10)
            st.caption(f"Silhouette terbaik pada K = {res['best_k']} (berdasarkan data yang sedang kamu load).")
        else:
            st.info("Dataset terlalu kecil untuk evaluasi silhouette yang stabil.")
    except Exception as e:
        st.warning(f"Gagal membuat grafik Elbow & Silhouette: {e}")

    st.subheader("Visualization in PCA Space (grafik notebook)")
    st.write(
        "Scatter plot ini memvisualisasikan data pada **2 komponen PCA pertama** dan diwarnai berdasarkan cluster. "
        "Tujuannya melihat apakah cluster terpisah dengan jelas atau tumpang tindih."
    )
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
                "Cluster pada Data yang Direduksi dengan PCA",
                "PCA Component 1",
                "PCA Component 2",
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

        st.subheader("Pipeline Output")
        st.write("Scaled features:")
        st.dataframe(pd.DataFrame([X_scaled[0]], columns=feature_names), use_container_width=True)
        st.write("PCA vector:")
        st.dataframe(pd.DataFrame([X_pca[0]], columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]), use_container_width=True)

elif page == "About":
    st.header("Tentang Saya")
    st.image("photos/foto-profil.jpg", width=150)
    
    st.write("""
    **Khoiriya Latifah**
    
    Saya adalah Dosen Informatika di Universitas Persatuan Guru Republik Indonesia Semarang (UPGRIS) dengan minat dan keahlian pada bidang Artificial Intelligence, Machine Learning, Data Mining, dan Sistem Informasi. Saya aktif dalam kegiatan pengajaran, penelitian, dan pengabdian kepada masyarakat, serta berfokus pada penerapan metode komputasi cerdas untuk menyelesaikan permasalahan nyata di bidang teknologi dan pendidikan.

    Dalam kegiatan akademik dan riset, saya berpengalaman mengimplementasikan berbagai algoritma seperti Random Forest, K-Means Clustering, dan metode analitik data lainnya untuk pengembangan sistem berbasis data dan website. Saya juga terlibat dalam publikasi ilmiah serta kolaborasi riset di tingkat nasional dan internasional.

    Keikutsertaan saya dalam program riset internasional Naveen Jindal Research Fellowship menunjukkan komitmen saya terhadap kolaborasi global dan pengembangan solusi inovatif di era digital. Pengalaman tersebut memperkuat perspektif saya dalam riset lintas budaya serta penerapan teknologi informasi pada konteks global.

    Saya memiliki komitmen untuk terus mengembangkan kompetensi profesional, mengikuti perkembangan teknologi terkini, serta membimbing mahasiswa agar siap menghadapi tantangan di era transformasi digital. Saya terbuka untuk kolaborasi riset, pengembangan proyek teknologi, serta jejaring akademik dan industri.
    """)
    st.markdown("---")

