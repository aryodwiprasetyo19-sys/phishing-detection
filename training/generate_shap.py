"""
training/generate_shap.py
=========================
Membuat objek SHAP explainer untuk setiap model dan menyimpannya ke
models_store/ sebagai file .pkl.

Mengapa TreeExplainer:
  Keempat algoritma (Random Forest, XGBoost, LightGBM, CatBoost) semuanya
  berbasis pohon keputusan (tree-based ensemble). shap.TreeExplainer bersifat
  eksak dan sangat efisien untuk model jenis ini, sehingga bisa dipakai
  seragam untuk keempatnya tanpa perlu pendekatan approximate yang lambat.

Explainer dibuat TANPA data background (mode "tree_path_dependent", metode
TreeSHAP asli). Mode ini tidak memerlukan sampel latar belakang sama sekali.
Catatan: memberi `data=` (mode "interventional") pada RandomForestClassifier
memicu crash native (access violation) pada kombinasi scikit-learn 1.5.0 +
shap 0.45.1 di Windows, sehingga dihindari untuk semua model demi konsistensi
dan stabilitas.

Menjalankan (setelah train_models.py):
    python training/generate_shap.py
"""

import sys
from pathlib import Path

import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    MODEL_NAMES,
    explainer_path,
    model_path,
)


def generate_all():
    """Buat & simpan SHAP explainer untuk semua model yang tersedia."""
    import shap  # diimpor lokal agar file lain tak wajib punya shap terpasang

    for name in MODEL_NAMES:
        mpath = model_path(name)
        if not mpath.exists():
            print(f"[shap] Lewati {name}: model belum dilatih.")
            continue

        model = joblib.load(mpath)
        print(f"[shap] Membuat TreeExplainer untuk {MODEL_NAMES[name]} ...")
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        joblib.dump(explainer, explainer_path(name))
        print(f"[shap] Explainer disimpan -> {explainer_path(name)}")

    print("[shap] Semua explainer berhasil dibuat.")


if __name__ == "__main__":
    generate_all()
    print("[shap] Selesai. Model & explainer siap dipakai aplikasi web (app.py).")
