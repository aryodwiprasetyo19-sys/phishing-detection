"""
ml/model_loader.py
===================
Memuat seluruh artifact ML (model, scaler, SHAP explainer) SEKALI ke memori
saat aplikasi dimulai, lalu menyediakannya untuk dipakai ulang di setiap
request tanpa membaca disk berulang kali.

Pola yang dipakai: singleton sederhana berbasis variabel modul-level.
Cukup untuk skala skripsi dan tetap efisien (hemat I/O, latensi rendah).

Model & explainer dimuat secara "lazy-tolerant": bila sebuah file tidak ada
(mis. hanya sebagian model yang sudah dilatih), model tersebut dilewati dengan
peringatan, bukan membuat aplikasi gagal start. Ini membuat aplikasi tetap
bisa berjalan meski pipeline training belum lengkap dijalankan.
"""

import json

import joblib

from config import (
    MODEL_METRICS_PATH,
    MODEL_NAMES,
    SCALER_PATH,
    explainer_path,
    model_path,
)

# Cache modul-level (diisi sekali oleh load_all()).
_MODELS = {}
_EXPLAINERS = {}
_SCALER = None
_METRICS = {}
_METRICS_MTIME = None
_LOADED = False


def load_all(verbose: bool = True):
    """Muat semua artifact ke memori. Idempoten: aman dipanggil berkali-kali."""
    global _SCALER, _METRICS, _METRICS_MTIME, _LOADED

    if _LOADED:
        return

    # 1. Scaler
    if SCALER_PATH.exists():
        _SCALER = joblib.load(SCALER_PATH)
        if verbose:
            print(f"[model_loader] Scaler dimuat: {SCALER_PATH.name}")
    else:
        if verbose:
            print(f"[model_loader] PERINGATAN: scaler tidak ditemukan ({SCALER_PATH}).")

    # 2. Model & explainer per algoritma
    for name in MODEL_NAMES:
        mpath = model_path(name)
        if mpath.exists():
            _MODELS[name] = joblib.load(mpath)
            if verbose:
                print(f"[model_loader] Model dimuat: {name}")
        elif verbose:
            print(f"[model_loader] Lewati model '{name}' (belum dilatih).")

        epath = explainer_path(name)
        if epath.exists():
            _EXPLAINERS[name] = joblib.load(epath)
            if verbose:
                print(f"[model_loader] Explainer dimuat: {name}")

    # 3. Metrik evaluasi (untuk halaman perbandingan)
    if MODEL_METRICS_PATH.exists():
        with open(MODEL_METRICS_PATH, encoding="utf-8") as f:
            _METRICS = json.load(f)
        _METRICS_MTIME = MODEL_METRICS_PATH.stat().st_mtime
        if verbose:
            print(f"[model_loader] Metrik dimuat: {len(_METRICS)} model.")

    _LOADED = True
    if verbose:
        print(f"[model_loader] Selesai. Model aktif: {list(_MODELS.keys())}")


def get_model(name: str):
    """Ambil objek model dari memori."""
    if not _LOADED:
        load_all(verbose=False)
    if name not in _MODELS:
        raise KeyError(
            f"Model '{name}' tidak tersedia. Model aktif: {list(_MODELS.keys())}. "
            "Jalankan pipeline training terlebih dahulu."
        )
    return _MODELS[name]


def get_explainer(name: str):
    """Ambil SHAP explainer dari memori (None bila belum tersedia)."""
    if not _LOADED:
        load_all(verbose=False)
    return _EXPLAINERS.get(name)


def get_scaler():
    """Ambil scaler dari memori."""
    if not _LOADED:
        load_all(verbose=False)
    return _SCALER


def _reload_metrics_if_changed():
    """Baca ulang model_metrics.json bila file berubah sejak terakhir dimuat
    (mis. setelah retraining di notebook), tanpa perlu restart aplikasi.
    Hanya untuk file metrik (kecil) -- model/scaler/explainer tetap singleton."""
    global _METRICS, _METRICS_MTIME

    if not MODEL_METRICS_PATH.exists():
        return

    mtime = MODEL_METRICS_PATH.stat().st_mtime
    if mtime != _METRICS_MTIME:
        with open(MODEL_METRICS_PATH, encoding="utf-8") as f:
            _METRICS = json.load(f)
        _METRICS_MTIME = mtime


def get_metrics() -> dict:
    """Ambil dict metrik evaluasi semua model."""
    if not _LOADED:
        load_all(verbose=False)
    _reload_metrics_if_changed()
    return _METRICS


def best_model():
    """Kembalikan metrik model dengan F1-Score tertinggi, atau None bila
    metrik evaluasi belum tersedia. Dipakai untuk memberi tahu pengguna
    algoritma mana yang performanya terbaik menurut evaluasi historis."""
    metrics = get_metrics()
    if not metrics:
        return None
    return max(metrics.values(), key=lambda m: m.get("f1_score", 0))


def available_models() -> list:
    """Daftar nama model yang berhasil dimuat ke memori."""
    if not _LOADED:
        load_all(verbose=False)
    return list(_MODELS.keys())
