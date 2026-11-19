# Copyright (c) 2026 Aryo Dwi Prasetyo - Universitas Gunadarma. All Rights Reserved.
"""
database/models.py
==================
Fungsi akses data (query helper) untuk kedua tabel aplikasi:
  - predictions   : riwayat setiap prediksi yang dilakukan pengguna.
  - model_metrics  : metrik evaluasi tiap algoritma (untuk halaman perbandingan).

Menggunakan query berparameter (%s) untuk mencegah SQL injection. JSON
(fitur & nilai SHAP) disimpan sebagai kolom bertipe JSON pada MySQL.
"""

import json

from database.db import get_connection


def insert_prediction(result: dict) -> int:
    """Simpan satu hasil prediksi ke tabel predictions.

    Parameters
    ----------
    result : dict
        Keluaran dari ml.predictor.predict().

    Returns
    -------
    int : id baris yang baru dibuat.
    """
    sql = """
        INSERT INTO predictions
            (url, model_used, prediction_label, phishing_probability,
             risk_level, extracted_features, shap_values, suggestion,
             domain_age_days)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    params = (
        result["url"][:2048],
        result["model_used"],
        result["prediction_label"],
        result["phishing_probability"],
        result["risk_level"],
        json.dumps(result["extracted_features"], ensure_ascii=False),
        json.dumps(result["shap_values"], ensure_ascii=False),
        result["suggestion"],
        result.get("domain_age_days"),
    )
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.lastrowid


def get_all_predictions(limit: int = 50) -> list:
    """Ambil riwayat prediksi terbaru (default 50 baris)."""
    sql = """
        SELECT id, url, model_used, prediction_label, phishing_probability,
               risk_level, created_at
        FROM predictions
        ORDER BY created_at DESC, id DESC
        LIMIT %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            return cur.fetchall()


def get_prediction_by_id(prediction_id: int):
    """Ambil satu prediksi lengkap (termasuk fitur & SHAP) berdasarkan id."""
    sql = "SELECT * FROM predictions WHERE id = %s"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (prediction_id,))
            row = cur.fetchone()
    if row:
        # Kolom JSON dikembalikan sebagai string oleh sebagian versi driver;
        # pastikan menjadi dict Python.
        for key in ("extracted_features", "shap_values"):
            if isinstance(row.get(key), str):
                row[key] = json.loads(row[key])
    return row


def clear_predictions():
    """Hapus seluruh riwayat prediksi (reset tabel predictions)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM predictions")


def count_predictions() -> int:
    """Total jumlah prediksi tersimpan (untuk statistik di Home)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total FROM predictions")
            return cur.fetchone()["total"]


def save_model_metrics(metrics: dict):
    """Simpan/segarkan metrik evaluasi model ke tabel model_metrics.

    Isi tabel dikosongkan lebih dulu agar selalu mencerminkan hasil training
    terakhir (bukan menumpuk versi lama).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM model_metrics")
            sql = """
                INSERT INTO model_metrics
                    (model_name, accuracy, precision_score, recall_score,
                     f1_score, roc_auc, training_time_sec)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            for m in metrics.values():
                cur.execute(
                    sql,
                    (
                        m["model_name"],
                        m["accuracy"],
                        m["precision"],
                        m["recall"],
                        m["f1_score"],
                        m["roc_auc"],
                        m.get("training_time_sec", 0.0),
                    ),
                )


def get_model_metrics() -> list:
    """Ambil metrik semua model dari DB (untuk halaman perbandingan)."""
    sql = """
        SELECT model_name, accuracy, precision_score, recall_score,
               f1_score, roc_auc, training_time_sec, trained_at
        FROM model_metrics
        ORDER BY f1_score DESC
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
