"""
routes/main_routes.py
=====================
Blueprint untuk halaman umum: Home ('/') dan Tentang ('/about').
"""

from flask import Blueprint, render_template

from config import MODEL_NAMES
from ml import model_loader

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def home():
    """Halaman utama: penjelasan singkat + tautan ke deteksi.

    Menampilkan jumlah prediksi tersimpan bila database aktif; bila DB tidak
    tersedia, halaman tetap tampil tanpa statistik (tidak menggagalkan render).
    """
    total_predictions = None
    try:
        from database.models import count_predictions

        total_predictions = count_predictions()
    except Exception:
        total_predictions = None

    return render_template(
        "home.html",
        total_predictions=total_predictions,
        available_models=model_loader.available_models(),
        model_names=MODEL_NAMES,
    )


@main_bp.route("/about")
def about():
    """Halaman Tentang: metodologi penelitian, fitur, dan algoritma."""
    return render_template("about.html", model_names=MODEL_NAMES)
