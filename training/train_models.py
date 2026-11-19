# Copyright (c) 2026 Aryo Dwi Prasetyo - Universitas Gunadarma. All Rights Reserved.
"""
training/train_models.py
========================
Melatih keempat algoritma (Random Forest, XGBoost, LightGBM, CatBoost) pada
data yang sama, lalu menyimpan tiap model beserta scaler ke models_store/.

Desain:
  - Semua model tree-based sehingga tidak wajib melakukan scaling. Namun scaler
    (StandardScaler) tetap dilatih dan disimpan agar pipeline inference seragam
    dan mudah diperluas bila kelak menambah model berbasis jarak/linear.
  - Setiap model memakai hyperparameter yang ringan-namun-wajar agar training
    cepat saat demo, tetapi tetap memberi hasil kompetitif.
  - Waktu training tiap model dicatat untuk ditampilkan pada perbandingan.

Menjalankan:
    python training/train_models.py --positive-label 1
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import MODEL_NAMES, MODELS_STORE_DIR, SCALER_PATH, model_path  # noqa: E402
from training.preprocessing import load_and_extract, split_data  # noqa: E402

RANDOM_STATE = 42


def _build_models(overrides: dict = None):
    """Membangun dict {nama: estimator} untuk keempat algoritma.

    Import pustaka boosting dilakukan di dalam fungsi agar file preprocessing
    tetap bisa dijalankan meski xgboost/lightgbm/catboost belum terpasang.

    Parameters
    ----------
    overrides : dict, opsional
        {nama_model: {hyperparameter: nilai}} -- hasil pencarian
        `training/tune_models.py` (lihat `load_best_params()`). Nilai default
        di bawah tetap dipakai untuk hyperparameter yang tidak di-override.
    """
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    overrides = overrides or {}

    rf_kwargs = dict(
        n_estimators=200,
        max_depth=45,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf_kwargs.update(overrides.get("random_forest", {}))

    xgb_kwargs = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    xgb_kwargs.update(overrides.get("xgboost", {}))

    lgb_kwargs = dict(
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.1,
        num_leaves=31,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    lgb_kwargs.update(overrides.get("lightgbm", {}))

    cat_kwargs = dict(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        random_seed=RANDOM_STATE,
        verbose=False,
    )
    cat_kwargs.update(overrides.get("catboost", {}))

    models = {
        "random_forest": RandomForestClassifier(**rf_kwargs),
        "xgboost": XGBClassifier(**xgb_kwargs),
        "lightgbm": LGBMClassifier(**lgb_kwargs),
        "catboost": CatBoostClassifier(**cat_kwargs),
    }
    return models


def train_all(use_processed=False, param_overrides: dict = None):
    """Latih seluruh model dan simpan artifact.

    Parameters
    ----------
    use_processed : bool
        Pakai data bersih tersimpan (`dataset/processed/`) alih-alih
        mengekstraksi ulang dari `dataset/raw/`.
    param_overrides : dict, opsional
        Hyperparameter hasil tuning per model (lihat `_build_models`). Bila
        None, dipakai hyperparameter default yang sudah teruji.

    Mengembalikan tuple (X_train, X_test, y_train, y_test, feature_names,
    training_times) untuk dipakai kembali oleh evaluate_models & generate_shap.
    """
    MODELS_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data (gabungan semua dataset di dataset/raw/).
    #    use_processed=True memakai data bersih yang sudah tersimpan (lebih cepat).
    if use_processed:
        from training.preprocessing import load_processed
        X, y, feature_names = load_processed()
    else:
        X, y, feature_names = load_and_extract()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 2. Scaler (dilatih pada data train saja untuk mencegah kebocoran data)
    scaler = StandardScaler()
    # Fit pada array numpy (tanpa nama kolom) agar konsisten dengan inference
    # yang juga mengirim array numpy -> menghindari peringatan feature-name sklearn.
    X_train_scaled = scaler.fit_transform(X_train.to_numpy())
    joblib.dump(scaler, SCALER_PATH)
    print(f"[train] Scaler disimpan ke: {SCALER_PATH}")

    # 3. Latih tiap model
    models = _build_models(param_overrides)
    training_times = {}

    for name, model in models.items():
        display = MODEL_NAMES.get(name, name)
        print(f"[train] Melatih {display} ...")
        start = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        elapsed = time.perf_counter() - start
        training_times[name] = round(elapsed, 4)

        joblib.dump(model, model_path(name))
        print(f"[train] {display} selesai ({elapsed:.2f}s) -> {model_path(name)}")

    print("[train] Semua model berhasil dilatih dan disimpan.")

    # Simpan data test ke disk agar tahap evaluasi & SHAP tidak perlu
    # mengulang preprocessing dari awal.
    joblib.dump(
        {
            "X_test": X_test,
            "y_test": y_test,
            "X_train": X_train,
            "feature_names": feature_names,
            "training_times": training_times,
        },
        MODELS_STORE_DIR / "_train_artifacts.pkl",
    )

    return X_train, X_test, y_train, y_test, feature_names, training_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training 4 model deteksi phishing.")
    parser.add_argument(
        "--use-processed",
        action="store_true",
        help="Pakai data bersih tersimpan (lewati ekstraksi ulang).",
    )
    parser.add_argument(
        "--tuned",
        action="store_true",
        help="Pakai hyperparameter hasil training/tune_models.py bila tersedia.",
    )
    args = parser.parse_args()

    overrides = None
    if args.tuned:
        from training.tune_models import load_best_params
        overrides = load_best_params()
        if not overrides:
            print("[train] --tuned diminta tapi best_params.json belum ada -- "
                  "jalankan training/tune_models.py dulu. Memakai default.")

    train_all(use_processed=args.use_processed, param_overrides=overrides)
    print("[train] Selesai. Lanjutkan dengan: python training/evaluate_models.py")
