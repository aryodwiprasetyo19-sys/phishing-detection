# Copyright (c) 2026 Aryo Dwi Prasetyo - Universitas Gunadarma. All Rights Reserved.
"""
routes/history_routes.py
========================
Blueprint untuk halaman Riwayat Prediksi ('/history') dan Perbandingan
Algoritma ('/comparison').

Sumber data perbandingan: tabel model_metrics (bila DB aktif) atau langsung
dari model_metrics.json (fallback). Ini membuat halaman tetap berfungsi
walau database tidak dikonfigurasi.
"""

from flask import Blueprint, flash, redirect, render_template, url_for

from config import Config, MODEL_NAMES
from ml import model_loader

history_bp = Blueprint("history", __name__)


def _describe_best_model(metrics: list):
    """Susun kalimat penjelasan kenapa model teratas (F1-Score tertinggi)
    terpilih, dengan membandingkan angka aktualnya terhadap model lain --
    bukan sekadar menyebut "F1 tertinggi" tanpa konteks."""
    if not metrics:
        return None
    best = metrics[0]
    others = metrics[1:]
    if not others:
        return (
            f"{best['display_name']} menjadi satu-satunya model yang tersedia "
            f"(F1-Score {best['f1_score']:.4f})."
        )

    runner_up = others[0]
    gap = best["f1_score"] - runner_up["f1_score"]

    sentence = (
        f"{best['display_name']} terpilih sebagai model terbaik karena F1-Score-nya "
        f"tertinggi ({best['f1_score']:.4f}), unggul {gap:.4f} poin dari "
        f"{runner_up['display_name']} ({runner_up['f1_score']:.4f}). Keunggulan ini "
        f"didukung kombinasi Precision {best['precision']:.4f} dan Recall "
        f"{best['recall']:.4f} yang seimbang -- precision tinggi berarti jarang salah "
        f"menandai situs aman sebagai phishing, recall tinggi berarti jarang melewatkan "
        f"situs phishing sungguhan."
    )

    timed = [m for m in metrics if m.get("training_time_sec") is not None]
    if timed:
        fastest = min(timed, key=lambda m: m["training_time_sec"])
        if fastest["model_name"] != best["model_name"] and best.get("training_time_sec"):
            sentence += (
                f" Sebagai catatan, waktu latih {best['display_name']} "
                f"({best['training_time_sec']:.2f} detik) lebih lama dibanding "
                f"{fastest['display_name']} ({fastest['training_time_sec']:.2f} detik), "
                f"namun akurasi klasifikasi lebih diprioritaskan pada sistem ini."
            )
    return sentence


@history_bp.route("/history")
def history():
    """Menampilkan daftar prediksi terbaru dari database."""
    predictions = []
    db_available = True
    try:
        from database.models import get_all_predictions

        predictions = get_all_predictions(limit=Config.HISTORY_PAGE_SIZE)
    except Exception:
        db_available = False

    return render_template(
        "history.html",
        predictions=predictions,
        db_available=db_available,
        model_names=MODEL_NAMES,
    )


@history_bp.route("/history/reset", methods=["POST"])
def reset_history():
    """Kosongkan seluruh riwayat prediksi (tabel predictions)."""
    try:
        from database.models import clear_predictions

        clear_predictions()
        flash("Riwayat prediksi berhasil direset.", "success")
    except Exception as exc:  # noqa: BLE001
        flash(f"Gagal mereset riwayat: {exc}", "danger")

    return redirect(url_for("history.history"))


@history_bp.route("/comparison")
def comparison():
    """Menampilkan tabel & grafik perbandingan metrik keempat algoritma."""
    metrics = []

    # Prioritas 1: dari database (historis).
    try:
        from database.models import get_model_metrics

        rows = get_model_metrics()
        for row in rows:
            metrics.append(
                {
                    "model_name": row["model_name"],
                    "display_name": MODEL_NAMES.get(row["model_name"], row["model_name"]),
                    "accuracy": row["accuracy"],
                    "precision": row["precision_score"],
                    "recall": row["recall_score"],
                    "f1_score": row["f1_score"],
                    "roc_auc": row["roc_auc"],
                    "training_time_sec": row.get("training_time_sec"),
                }
            )
    except Exception:
        metrics = []

    # Prioritas 2 (fallback): dari model_metrics.json yang sudah dimuat ke memori.
    if not metrics:
        raw = model_loader.get_metrics()
        for m in raw.values():
            metrics.append(
                {
                    "model_name": m["model_name"],
                    "display_name": m.get("display_name", m["model_name"]),
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1_score": m["f1_score"],
                    "roc_auc": m["roc_auc"],
                    "training_time_sec": m.get("training_time_sec"),
                }
            )

    # Urutkan berdasarkan F1-score menurun; tandai yang terbaik.
    metrics.sort(key=lambda m: m["f1_score"], reverse=True)
    best_model = metrics[0]["model_name"] if metrics else None
    best_explanation = _describe_best_model(metrics)

    return render_template(
        "comparison.html",
        metrics=metrics,
        best_model=best_model,
        best_explanation=best_explanation,
    )
