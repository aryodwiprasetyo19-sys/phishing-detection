"""
config.py
=========
Konfigurasi terpusat untuk seluruh aplikasi (web maupun pipeline training).

Semua path, konstanta, dan pengaturan diletakkan di sini agar tidak tersebar di
banyak file (Single Source of Truth). Nilai sensitif (kredensial DB, secret key)
diambil dari environment variable via python-dotenv sehingga tidak perlu
di-hardcode ke dalam kode sumber.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Muat variabel dari file .env (jika ada) ke environment
load_dotenv()

# ---------------------------------------------------------------------------
# PATH DASAR
# ---------------------------------------------------------------------------
# BASE_DIR = folder root project (lokasi file config.py ini berada)
BASE_DIR = Path(__file__).resolve().parent

MODELS_STORE_DIR = BASE_DIR / "models_store"
DATASET_RAW_DIR = BASE_DIR / "dataset" / "raw"
DATASET_PROCESSED_DIR = BASE_DIR / "dataset" / "processed"

# File dataset default
RAW_DATASET_PATH = DATASET_RAW_DIR / "phishing_dataset.csv"
PROCESSED_DATASET_PATH = DATASET_PROCESSED_DIR / "phishing_dataset_clean.csv"

# File metrik hasil evaluasi (dibaca halaman "Perbandingan Algoritma")
MODEL_METRICS_PATH = MODELS_STORE_DIR / "model_metrics.json"

# Scaler bersama untuk semua model
SCALER_PATH = MODELS_STORE_DIR / "scaler.pkl"


# ---------------------------------------------------------------------------
# DAFTAR MODEL YANG DIBANDINGKAN
# ---------------------------------------------------------------------------
# Kunci = nama internal (dipakai di path file, DB, dan pemilihan model).
# Nilai  = nama tampilan yang ramah untuk ditampilkan di antarmuka.
MODEL_NAMES = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}

# Model default bila pengguna tidak memilih algoritma tertentu.
DEFAULT_MODEL = "random_forest"


def model_path(model_name: str) -> Path:
    """Path file .pkl untuk sebuah model."""
    return MODELS_STORE_DIR / f"{model_name}.pkl"


def explainer_path(model_name: str) -> Path:
    """Path file .pkl untuk SHAP explainer sebuah model."""
    return MODELS_STORE_DIR / f"shap_explainer_{model_name}.pkl"


# ---------------------------------------------------------------------------
# THRESHOLD RISK LEVEL
# ---------------------------------------------------------------------------
# Probabilitas phishing (0.0 - 1.0) dipetakan ke tiga tingkat risiko.
#   < 0.30              -> Low
#   0.30 <= p < 0.70    -> Medium
#   >= 0.70             -> High
RISK_THRESHOLD_LOW = 0.30
RISK_THRESHOLD_HIGH = 0.70

# Ambang keputusan label akhir (phishing vs legitimate).
CLASSIFICATION_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# KONFIGURASI DATABASE MySQL
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "phishing_detector_db"),
    "charset": "utf8mb4",
}


# ---------------------------------------------------------------------------
# KONFIGURASI FLASK
# ---------------------------------------------------------------------------
class Config:
    """Konfigurasi aplikasi Flask."""

    SECRET_KEY = os.getenv("SECRET_KEY", "ganti-dengan-secret-key-acak-di-produksi")
    # Batas panjang URL yang boleh diproses (mencegah input abnormal).
    MAX_URL_LENGTH = 2048
    # Jumlah baris riwayat yang ditampilkan pada halaman History.
    HISTORY_PAGE_SIZE = 50
    JSON_SORT_KEYS = False
