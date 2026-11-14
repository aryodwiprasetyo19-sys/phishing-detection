"""
training/preprocessing.py
=========================
Mengubah SATU ATAU BANYAK dataset URL mentah menjadi satu data siap latih.

Alur:
  1. Pindai folder dataset/raw/ -> ambil SEMUA file .csv (nama bebas).
  2. Untuk tiap file: deteksi kolom URL & label, normalisasi label ke
     {1: phishing, 0: legitimate}.
  3. Gabungkan seluruh file, buang duplikat URL.
  4. Ekstrak 40 fitur numerik dari tiap URL (URLFeatureExtractor).
  5. Simpan SATU data bersih ke dataset/processed/ (fitur + label).
  6. Sediakan split train/test stratified.

KONVENSI LABEL (eksplisit agar tidak ambigu antar-dataset):
  phishing -> 1 ; legitimate -> 0

Label TEKS ("phishing"/"legitimate"/"benign"/...) dipetakan otomatis via kata.
Label ANGKA berbeda konvensi antar-dataset (mis. sebagian memakai 1=legitimate).
Karena itu setiap file berlabel angka membutuhkan info "nilai mana = phishing".
Sediakan file opsional dataset/raw/datasets.json:

    [
      {"file": "phiusiil.csv", "phishing_value": 0},
      {"file": "victusadi.csv", "phishing_value": 1},
      {"file": "harisudhan.csv", "label_col": "status", "phishing_value": 0}
    ]

File yang tidak terdaftar & berlabel angka akan memakai default phishing_value=1
disertai PERINGATAN agar Anda dapat mengoreksi bila konvensinya berbeda.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import DATASET_RAW_DIR, PROCESSED_DATASET_PATH  # noqa: E402
from ml.feature_extractor import URLFeatureExtractor  # noqa: E402

_URL_COLUMN_CANDIDATES = ["url", "urls", "domain", "website", "link", "uri"]
_LABEL_COLUMN_CANDIDATES = [
    "label", "status", "type", "result", "class", "target", "is_phishing", "phishing",
]
_PHISHING_WORDS = {"phishing", "phish", "malicious", "bad", "spam", "fraud", "yes"}
_LEGIT_WORDS = {"legitimate", "legit", "benign", "good", "safe", "no", "ham", "normal"}


def _find_column(df, candidates, fallback_index=None):
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    if fallback_index is not None and len(df.columns) > fallback_index:
        return df.columns[fallback_index]
    return None


def _load_manifest(raw_dir: Path) -> dict:
    """Muat datasets.json (opsional) -> dict {nama_file: entry}."""
    manifest_path = raw_dir / "datasets.json"
    if not manifest_path.exists():
        return {}
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {e["file"]: e for e in entries if "file" in e}
    except Exception as exc:  # noqa: BLE001
        print(f"[preprocessing] Gagal membaca datasets.json: {exc}")
        return {}


def _normalize_series(labels: pd.Series, phishing_value) -> pd.Series:
    """Ubah kolom label menjadi 1 (phishing) / 0 (legitimate). NaN utk tak dikenal."""
    def _one(v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in _PHISHING_WORDS:
                return 1
            if s in _LEGIT_WORDS:
                return 0
            # Teks berupa angka ("0"/"1")
            try:
                return 1 if float(s) == float(phishing_value) else 0
            except ValueError:
                return None
        try:
            return 1 if float(v) == float(phishing_value) else 0
        except (TypeError, ValueError):
            return None

    return labels.apply(_one)


def _read_one_file(path: Path, manifest: dict) -> pd.DataFrame:
    """Baca satu file -> DataFrame kolom ['url','label'] (label 1/0)."""
    entry = manifest.get(path.name, {})
    df = pd.read_csv(path)

    url_col = entry.get("url_col") or _find_column(df, _URL_COLUMN_CANDIDATES, fallback_index=0)
    label_col = entry.get("label_col") or _find_column(df, _LABEL_COLUMN_CANDIDATES, fallback_index=1)
    if url_col is None or label_col is None:
        print(f"[preprocessing]  ! Lewati {path.name}: kolom URL/label tak terdeteksi.")
        return pd.DataFrame(columns=["url", "label"])

    sample = df[label_col].dropna().head(50)
    is_numeric = pd.to_numeric(sample, errors="coerce").notna().all()
    phishing_value = entry.get("phishing_value", 1)
    if is_numeric and "phishing_value" not in entry:
        print(
            f"[preprocessing]  * {path.name}: label ANGKA tanpa konfigurasi -> "
            f"asumsi {phishing_value}=phishing. Jika keliru, tambahkan entri di "
            f"dataset/raw/datasets.json."
        )

    out = pd.DataFrame(
        {
            "url": df[url_col].astype(str).str.strip(),
            "label": _normalize_series(df[label_col], phishing_value),
        }
    )
    out = out[(out["url"] != "") & out["label"].notna()]
    out["label"] = out["label"].astype(int)
    print(f"[preprocessing]  - {path.name}: {len(out)} baris valid "
          f"(url='{url_col}', label='{label_col}')")
    return out


def load_all_raw(raw_dir: Path = DATASET_RAW_DIR) -> pd.DataFrame:
    """Gabungkan semua CSV di folder mentah -> DataFrame unik ['url','label']."""
    raw_dir = Path(raw_dir)
    files = sorted(p for p in raw_dir.glob("*.csv") if p.name != "datasets.json")
    if not files:
        raise FileNotFoundError(
            f"Tidak ada file .csv di {raw_dir}. Letakkan satu atau beberapa "
            "dataset URL berlabel di folder tersebut."
        )
    manifest = _load_manifest(raw_dir)
    print(f"[preprocessing] Menemukan {len(files)} file dataset di {raw_dir}")

    frames = [_read_one_file(p, manifest) for p in files]
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
    print(f"[preprocessing] Total {before} baris -> {len(combined)} setelah buang duplikat")
    return combined


def load_and_extract(raw_dir: Path = DATASET_RAW_DIR, save_processed: bool = True):
    """Muat semua dataset mentah, ekstrak fitur, kembalikan (X, y, feature_names)."""
    combined = load_all_raw(raw_dir)

    extractor = URLFeatureExtractor()
    feature_names = extractor.FEATURE_NAMES

    rows, total = [], len(combined)
    for i, url in enumerate(combined["url"].astype(str), start=1):
        rows.append(extractor.extract_ordered(url))
        if i % 5000 == 0 or i == total:
            print(f"[preprocessing] Ekstraksi fitur {i}/{total}")

    X = pd.DataFrame(rows, columns=feature_names)
    y = combined["label"].reset_index(drop=True)

    n_phish, n_legit = int((y == 1).sum()), int((y == 0).sum())
    print(f"[preprocessing] Distribusi -> phishing={n_phish}, legitimate={n_legit}")

    if save_processed:
        processed = X.copy()
        processed.insert(0, "url", combined["url"].values)
        processed["label"] = y
        Path(PROCESSED_DATASET_PATH).parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(PROCESSED_DATASET_PATH, index=False)
        print(f"[preprocessing] Data bersih disimpan: {PROCESSED_DATASET_PATH}")

    return X, y, feature_names


def load_processed():
    """Muat data bersih yang sudah disimpan -> (X, y, feature_names)."""
    if not Path(PROCESSED_DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Data bersih belum ada di {PROCESSED_DATASET_PATH}. "
            "Jalankan load_and_extract() lebih dulu."
        )
    df = pd.read_csv(PROCESSED_DATASET_PATH)
    feature_names = URLFeatureExtractor().FEATURE_NAMES
    X = df[feature_names]
    y = df["label"]
    return X, y, feature_names


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Bagi data train/test secara stratified."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[preprocessing] Split -> train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    argparse.ArgumentParser(description="Preprocessing multi-dataset phishing.").parse_args()
    X, y, names = load_and_extract()
    split_data(X, y)
    print("[preprocessing] Selesai.")
