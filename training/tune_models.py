"""
training/tune_models.py
========================
Pencarian hyperparameter (RandomizedSearchCV) untuk keempat algoritma.

Pencarian dijalankan pada SUBSAMPLE stratified dari data LATIH (bukan
seluruh ~776 ribu baris), lalu hyperparameter terbaik yang ditemukan dipakai
untuk melatih ulang model pada SELURUH data latih lewat
`train_models.train_all(param_overrides=...)`.

Kenapa subsample untuk pencarian (bukan data penuh):
  RandomizedSearchCV mengevaluasi banyak kombinasi hyperparameter x beberapa
  fold cross-validation. Random Forest saja butuh ±187 detik untuk SATU kali
  fit pada data latih penuh (lihat train_models.py) -- menjalankan puluhan
  fit semacam itu untuk pencarian akan memakan waktu berjam-jam. Mencari pada
  subsample stratified yang cukup besar (default 150.000 baris, ~19% data
  latih) jauh lebih cepat namun tetap representatif untuk menemukan wilayah
  hyperparameter yang baik, karena keempat algoritma bersifat tree-based dan
  perilaku relatifnya antar-kombinasi hyperparameter umumnya konsisten pada
  subsample besar vs data penuh.

Menjalankan:
    python training/tune_models.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import MODEL_NAMES, MODELS_STORE_DIR  # noqa: E402
from training.preprocessing import load_processed, split_data  # noqa: E402

RANDOM_STATE = 42
BEST_PARAMS_PATH = MODELS_STORE_DIR / "best_params.json"

# Ruang pencarian per algoritma -- dipusatkan di sekitar hyperparameter
# default yang sudah dipakai (train_models.py) agar pencarian tetap relevan
# dengan konfigurasi yang sudah terbukti wajar, sambil membuka kemungkinan
# nilai lebih besar/kecil untuk mengejar performa lebih baik.
PARAM_DISTRIBUTIONS = {
    "random_forest": {
        "n_estimators": [200, 300, 400, 500, 600],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_leaf": [1, 2, 4, 6],
        "min_samples_split": [2, 5, 10],
        # "max_features": None (semua fitur per split) SENGAJA dihapus dari
        # ruang pencarian -- pengujian sebelumnya menemukan kombinasi ini
        # meningkatkan waktu latih pada data penuh 8,6x (186,8s -> 1.611,9s)
        # untuk kenaikan F1 yang nyaris nol (+0,0007), jauh lebih mahal
        # dibanding gain-nya. "sqrt"/"log2" (subset fitur per split) adalah
        # perilaku standar Random Forest dan jauh lebih efisien.
        "max_features": ["sqrt", "log2"],
    },
    "xgboost": {
        "n_estimators": [200, 300, 400, 500, 600],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [0.5, 1, 1.5, 2],
    },
    "lightgbm": {
        "n_estimators": [200, 300, 400, 500, 600],
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.03, 0.05, 0.1, 0.15, 0.2],
        "feature_fraction": [0.7, 0.8, 0.9, 1.0],
        "bagging_fraction": [0.7, 0.8, 0.9, 1.0],
        "min_child_samples": [5, 10, 20, 30],
    },
    "catboost": {
        "iterations": [200, 300, 400, 500, 600],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.1, 0.15, 0.2],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
    },
}


def _search_estimator(name: str):
    """Estimator dasar untuk pencarian, dipaksa SATU thread.

    Paralelisme dipindahkan seluruhnya ke `RandomizedSearchCV(n_jobs=-1)`
    supaya banyak kombinasi hyperparameter dievaluasi BERSAMAAN di semua
    core, bukan satu kombinasi per satu yang masing-masing multi-thread --
    lebih efisien untuk pencarian dengan banyak kandidat kecil.
    """
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    builders = {
        "random_forest": lambda: RandomForestClassifier(n_jobs=1, random_state=RANDOM_STATE),
        "xgboost": lambda: XGBClassifier(
            eval_metric="logloss", n_jobs=1, random_state=RANDOM_STATE
        ),
        "lightgbm": lambda: LGBMClassifier(n_jobs=1, random_state=RANDOM_STATE, verbose=-1),
        "catboost": lambda: CatBoostClassifier(
            thread_count=1, random_seed=RANDOM_STATE, verbose=False
        ),
    }
    return builders[name]()


def tune_all(
    sample_size: int = 250_000,
    n_iter: int = 25,
    cv: int = 3,
    scoring: str = "f1",
    random_state: int = RANDOM_STATE,
) -> dict:
    """Cari hyperparameter terbaik tiap algoritma via RandomizedSearchCV.

    Subsample diambil dari X_train/y_train hasil `split_data()` dengan seed
    yang SAMA dengan yang dipakai `train_models.train_all()`, sehingga tidak
    ada baris data uji yang bocor ke proses pencarian.

    Returns
    -------
    dict {nama_model: {best_params, best_cv_f1, search_time_sec, top5}}
    """
    X, y, feature_names = load_processed()
    X_train, _, y_train, _ = split_data(X, y)

    if sample_size < len(X_train):
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train, train_size=sample_size,
            stratify=y_train, random_state=random_state,
        )
    else:
        X_sub, y_sub = X_train, y_train

    # Scaler khusus untuk pencarian (tidak disimpan/dipakai untuk inference --
    # scaler final tetap dilatih pada seluruh data latih oleh train_all()).
    search_scaler = StandardScaler()
    X_sub_scaled = search_scaler.fit_transform(X_sub.to_numpy())

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    results = {}
    MODELS_STORE_DIR.mkdir(parents=True, exist_ok=True)

    for name in MODEL_NAMES:
        display = MODEL_NAMES[name]
        print(
            f"[tune] Mencari hyperparameter {display} "
            f"({n_iter} kombinasi x {cv}-fold, subsample={len(X_sub):,} baris) ..."
        )
        start = time.perf_counter()

        search = RandomizedSearchCV(
            estimator=_search_estimator(name),
            param_distributions=PARAM_DISTRIBUTIONS[name],
            n_iter=n_iter,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
            refit=False,
            random_state=random_state,
        )
        search.fit(X_sub_scaled, y_sub)
        elapsed = time.perf_counter() - start

        order = np.argsort(search.cv_results_["mean_test_score"])[::-1][:5]
        top5 = [
            {
                "params": search.cv_results_["params"][i],
                "mean_f1": round(float(search.cv_results_["mean_test_score"][i]), 4),
            }
            for i in order
        ]

        results[name] = {
            "best_params": search.best_params_,
            "best_cv_f1": round(float(search.best_score_), 4),
            "search_time_sec": round(elapsed, 2),
            "top5": top5,
        }
        print(
            f"[tune] {display} selesai ({elapsed:.1f}s) -- "
            f"F1 CV terbaik={search.best_score_:.4f} -> {search.best_params_}"
        )

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[tune] Hyperparameter terbaik disimpan ke: {BEST_PARAMS_PATH}")
    return results


def load_best_params() -> dict:
    """Muat hyperparameter terbaik hasil `tune_all()`.

    Returns {} bila `best_params.json` belum ada (mis. belum pernah
    menjalankan tuning) -- pemanggil sebaiknya fallback ke default.
    """
    if not BEST_PARAMS_PATH.exists():
        return {}
    data = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    return {name: info["best_params"] for name, info in data.items()}


if __name__ == "__main__":
    tune_all()
    print("[tune] Selesai. Lanjutkan dengan: python training/train_models.py --use-processed --tuned")
