"""
# ==============================================================================
# PhishGuard XAI — Deteksi Website Phishing Berbasis Explainable Machine Learning
# Author      : Aryo Dwi Prasetyo (NPM: 50422283)
# Institution : Informatika, Fakultas Teknologi Industri, Universitas Gunadarma
# Advisor     : Dr. Lely Prananingrum, S.Kom., MMSI
# Defense     : 18 Agustus 2026
# Copyright   : (c) 2026 Aryo Dwi Prasetyo. All Rights Reserved.
# ==============================================================================
app.py
======
Entry point utama aplikasi web Flask (tanpa application-factory yang rumit,
cukup instansiasi langsung namun tetap modular lewat Blueprint).

Tanggung jawab:
  - Membuat objek Flask dan memuat konfigurasi.
  - Memuat seluruh artifact ML ke memori SEKALI saat startup (model_loader).
  - Mendaftarkan blueprint (main, detection, history).
  - Menyediakan error handler 404 & 500.

Menjalankan (development):
    python app.py
Menjalankan (production):
    gunicorn wsgi:app
"""

from flask import Flask, render_template

from config import Config
from ml import model_loader
from routes.detection_routes import detection_bp
from routes.history_routes import history_bp
from routes.main_routes import main_bp


def create_app() -> Flask:
    """Buat dan konfigurasikan aplikasi Flask.

    Fungsi ini dipakai baik oleh server pengembangan (python app.py) maupun
    oleh Gunicorn via wsgi.py, sehingga pemuatan model dilakukan di sini agar
    konsisten di kedua mode.
    """
    app = Flask(__name__)
    app.config.from_object(Config)

    # Muat model/scaler/explainer/metrik ke memori sekali saat startup.
    model_loader.load_all(verbose=True)

    # Registrasi blueprint.
    app.register_blueprint(main_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(history_bp)

    # Error handler.
    @app.errorhandler(404)
    def not_found(_error):
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def server_error(_error):
        return render_template("500.html"), 500

    return app


# Objek app tingkat modul agar bisa diimpor oleh wsgi.py.
app = create_app()


if __name__ == "__main__":
    print("=" * 60)
    print(" Phishing Detector XAI — Aplikasi Web")
    print(" Model aktif :", model_loader.available_models() or "(belum ada)")
    print(" URL         : http://127.0.0.1:5000  (atau http://<IP-lokal>:5000 dari HP)")
    print("=" * 60)
    # host=0.0.0.0 agar bisa diakses dari HP/perangkat lain di WiFi yang sama.
    # debug=True hanya untuk pengembangan. Di produksi gunakan Gunicorn.
    app.run(host="0.0.0.0", port=5000, debug=True)
