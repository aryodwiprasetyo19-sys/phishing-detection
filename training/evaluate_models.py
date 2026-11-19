# Copyright (c) 2026 Aryo Dwi Prasetyo - Universitas Gunadarma. All Rights Reserved.
"""
training/evaluate_models.py
===========================
Mengevaluasi keempat model pada data uji yang sama dan menyimpan hasil
perbandingan ke models_store/model_metrics.json.

Metrik yang dihitung (standar untuk klasifikasi biner):
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion Matrix (TN, FP, FN, TP)

Hasil ini menjadi sumber data halaman "Perbandingan Algoritma" pada web app,
dan juga di-insert ke tabel MySQL model_metrics (opsional, via init_db terpisah).

Menjalankan (setelah train_models.py):
    python training/evaluate_models.py
"""

import json
import sys
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    MODEL_METRICS_PATH,
    MODEL_NAMES,
    MODELS_STORE_DIR,
    SCALER_PATH,
    model_path,
)


def _load_test_data():
    """Muat data uji + scaler yang disimpan tahap training."""
    artifacts_path = MODELS_STORE_DIR / "_train_artifacts.pkl"
    if not artifacts_path.exists():
        raise FileNotFoundError(
            "Artefak training tidak ditemukan. Jalankan train_models.py terlebih dahulu."
        )
    data = joblib.load(artifacts_path)
    scaler = joblib.load(SCALER_PATH)
    return data, scaler


def evaluate_all():
    """Evaluasi semua model dan tulis model_metrics.json."""
    data, scaler = _load_test_data()
    X_test = data["X_test"]
    y_test = data["y_test"]
    training_times = data.get("training_times", {})

    X_test_scaled = scaler.transform(X_test.to_numpy())

    results = {}
    for name in MODEL_NAMES:
        mpath = model_path(name)
        if not mpath.exists():
            print(f"[evaluate] Lewati {name}: model belum dilatih ({mpath}).")
            continue

        model = joblib.load(mpath)
        y_pred = model.predict(X_test_scaled)
        # predict_proba[:, 1] = probabilitas kelas phishing (label 1).
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics = {
            "model_name": name,
            "display_name": MODEL_NAMES[name],
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
            "training_time_sec": training_times.get(name, 0.0),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }
        results[name] = metrics
        print(
            f"[evaluate] {MODEL_NAMES[name]:<14} "
            f"acc={metrics['accuracy']:.4f} f1={metrics['f1_score']:.4f} "
            f"auc={metrics['roc_auc']:.4f}"
        )

    if not results:
        print("[evaluate] Tidak ada model untuk dievaluasi.")
        return {}

    MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[evaluate] Metrik disimpan ke: {MODEL_METRICS_PATH}")

    # Sinkron ke tabel MySQL model_metrics bila DB aktif, supaya halaman
    # Perbandingan (yang memprioritaskan DB) tidak menampilkan hasil basi
    # dari eksekusi init_db.py sebelumnya. Gagal diam-diam bila DB belum
    # dikonfigurasi -- MySQL bersifat opsional untuk aplikasi ini.
    try:
        from database.models import save_model_metrics

        save_model_metrics(results)
        print("[evaluate] Metrik disinkronkan ke tabel MySQL model_metrics.")
    except Exception as exc:
        print(f"[evaluate] Lewati sinkronisasi DB (MySQL belum aktif?): {exc}")

    # Tentukan model terbaik berdasarkan F1-score.
    best = max(results.values(), key=lambda m: m["f1_score"])
    print(f"[evaluate] Model terbaik (F1): {best['display_name']} ({best['f1_score']})")
    return results


if __name__ == "__main__":
    evaluate_all()
    print("[evaluate] Selesai. Lanjutkan dengan: python training/generate_shap.py")
