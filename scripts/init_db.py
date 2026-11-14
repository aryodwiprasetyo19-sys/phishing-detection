"""
scripts/init_db.py
==================
Inisialisasi skema database MySQL untuk aplikasi.

Langkah:
  1. Membuat database (jika belum ada).
  2. Membuat tabel `predictions` dan `model_metrics`.
  3. (Opsional) memuat model_metrics.json hasil evaluasi ke tabel model_metrics.

Menjalankan:
    python scripts/init_db.py
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import DB_CONFIG, MODEL_METRICS_PATH  # noqa: E402
from database.db import get_connection  # noqa: E402

CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id                   INT AUTO_INCREMENT PRIMARY KEY,
    url                  VARCHAR(2048) NOT NULL,
    model_used           VARCHAR(50) NOT NULL,
    prediction_label     ENUM('legitimate', 'phishing') NOT NULL,
    phishing_probability FLOAT NOT NULL,
    risk_level           ENUM('Low', 'Medium', 'High') NOT NULL,
    extracted_features   JSON NOT NULL,
    shap_values          JSON NOT NULL,
    suggestion           TEXT,
    domain_age_days      INT NULL,
    created_at           DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

CREATE_MODEL_METRICS = """
CREATE TABLE IF NOT EXISTS model_metrics (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    model_name        VARCHAR(50) NOT NULL,
    accuracy          FLOAT NOT NULL,
    precision_score   FLOAT NOT NULL,
    recall_score      FLOAT NOT NULL,
    f1_score          FLOAT NOT NULL,
    roc_auc           FLOAT NOT NULL,
    training_time_sec FLOAT,
    trained_at        DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def create_database():
    """Membuat database bila belum ada (koneksi tanpa memilih DB)."""
    db_name = DB_CONFIG["database"]
    with get_connection(include_db=False) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    print(f"[init_db] Database '{db_name}' siap.")


def create_tables():
    """Membuat kedua tabel aplikasi."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_PREDICTIONS)
            cur.execute(CREATE_MODEL_METRICS)
    print("[init_db] Tabel 'predictions' & 'model_metrics' siap.")


def load_metrics_from_json():
    """Muat metrik dari model_metrics.json ke tabel (jika file tersedia)."""
    if not MODEL_METRICS_PATH.exists():
        print("[init_db] model_metrics.json belum ada, lewati pemuatan metrik.")
        return
    from database.models import save_model_metrics

    with open(MODEL_METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    save_model_metrics(metrics)
    print(f"[init_db] {len(metrics)} baris metrik dimuat ke tabel model_metrics.")


if __name__ == "__main__":
    create_database()
    create_tables()
    load_metrics_from_json()
    print("[init_db] Inisialisasi database selesai.")
