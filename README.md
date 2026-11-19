# PhishGuard XAI — Deteksi Website Phishing Berbasis Explainable Machine Learning

Aplikasi web untuk mendeteksi apakah sebuah URL berpotensi **phishing** atau **legitimate**,
lengkap dengan **penjelasan transparan (explainability)** menggunakan SHAP. Proyek ini membandingkan
empat algoritma ensemble: **Random Forest, XGBoost, LightGBM, dan CatBoost**.

### Informasi Riset & Penulis
- **Peneliti / Penulis:** Aryo Dwi Prasetyo
- **NPM:** 50422283
- **Program Studi:** Informatika
- **Fakultas:** Fakultas Teknologi Industri (FTI), Universitas Gunadarma
- **Dosen Pembimbing:** Dr. Lely Prananingrum, S.Kom., MMSI
- **Tanggal Sidang / Lulus:** 18 Agustus 2026
- **Judul Skripsi Resmi:**
  > *IMPLEMENTASI EXPLAINABLE MACHINE LEARNING UNTUK DETEKSI WEBSITE PHISHING BERBASIS WEB DENGAN PERBANDINGAN ALGORITMA RANDOM FOREST, XGBOOST, LIGHTGBM, DAN CATBOOST*

---

## Fitur Utama

- Ekstraksi **30 fitur leksikal & struktural** langsung dari teks URL (tanpa DNS/WHOIS/SSL/kunjungan situs).
- Perbandingan performa **4 algoritma** (accuracy, precision, recall, F1, ROC-AUC).
- **Explainability** dengan SHAP: kontribusi tiap fitur terhadap keputusan model.
- Tiga tingkat risiko: **Low (<30%)**, **Medium (30–70%)**, **High (>70%)**.
- Enam halaman: Home, Deteksi URL, Hasil Analisis, Perbandingan Algoritma, Riwayat Prediksi, Tentang.
- Riwayat prediksi tersimpan di **MySQL** (opsional — aplikasi tetap berjalan tanpa database).

---

## Teknologi

- **Backend:** Flask 3 (Blueprint), PyMySQL
- **Frontend:** Bootstrap 5, Chart.js, JavaScript (vanilla)
- **Machine Learning:** scikit-learn, XGBoost, LightGBM, CatBoost
- **Explainable AI:** SHAP (TreeExplainer)

---

## Struktur Proyek

```
phishing-detector-xai/
├── app.py                  # Entry point Flask (create_app + dev server)
├── wsgi.py                 # Entry point untuk Gunicorn (produksi)
├── config.py               # Konfigurasi terpusat (path, model, threshold, DB)
├── requirements.txt
├── .env.example            # Contoh variabel lingkungan
│
├── ml/                     # Inti machine learning
│   ├── feature_extractor.py    # Ekstraksi 30 fitur dari URL
│   ├── model_loader.py         # Cache model/scaler/explainer (singleton)
│   └── predictor.py            # Orkestrasi prediksi + SHAP + risk level
│
├── training/               # Pipeline pelatihan (dijalankan sekali di awal)
│   ├── preprocessing.py        # Load dataset + ekstraksi fitur + split
│   ├── train_models.py         # Latih 4 model + scaler
│   ├── evaluate_models.py      # Hitung metrik -> model_metrics.json
│   └── generate_shap.py        # Buat TreeExplainer per model
│
├── routes/                 # Blueprint Flask
│   ├── main_routes.py          # Home, Tentang
│   ├── detection_routes.py     # Form deteksi + API + halaman hasil
│   └── history_routes.py       # Riwayat & perbandingan algoritma
│
├── database/               # Lapisan database MySQL
│   ├── db.py                   # Koneksi (context manager)
│   └── models.py               # Query prediksi & metrik model
│
├── scripts/
│   └── init_db.py              # Buat database & tabel
│
├── utils/
│   └── validators.py           # Validasi & pembersihan input URL
│
├── static/                 # CSS & JS
│   ├── css/style.css
│   └── js/{shap_chart.js, detection.js}
│
├── templates/              # Halaman HTML (Jinja2)
├── models_store/           # Artefak model hasil training (.pkl, .json)
└── dataset/                # Dataset mentah & olahan
    ├── raw/
    └── processed/
```

---

## Persiapan

### 1. Instal dependensi

Disarankan menggunakan virtual environment.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Siapkan dataset

Unduh dataset publik berisi URL berlabel (kolom URL dan label phishing/legitimate),
misalnya dataset *"Phishing and Legitimate URLs"* dari Kaggle. Letakkan sebagai:

```
dataset/raw/phishing_dataset.csv
```

Skrip preprocessing mendeteksi kolom URL dan label secara otomatis. Konvensi label
yang digunakan: **1 = phishing**, **0 = legitimate**.

### 3. Konfigurasi lingkungan

```bash
cp .env.example .env
```

Sesuaikan kredensial MySQL dan `SECRET_KEY` di dalam `.env`.

---

## Menjalankan Pipeline Pelatihan

Jalankan berurutan dari direktori root proyek. Setiap langkah menghasilkan artefak
yang dipakai langkah berikutnya.

```bash
# 1. Preprocessing: ekstraksi fitur + split train/test
python -m training.preprocessing

# 2. Latih keempat model + scaler
python -m training.train_models

# 3. Evaluasi model -> models_store/model_metrics.json
python -m training.evaluate_models

# 4. Buat explainer SHAP per model
python -m training.generate_shap
```

Setelah selesai, folder `models_store/` akan berisi `*.pkl`, `scaler.pkl`,
`*_explainer.pkl`, dan `model_metrics.json`.

---

## Menyiapkan Database (opsional)

Riwayat prediksi dan metrik dapat disimpan di MySQL. Jika database tidak tersedia,
aplikasi tetap berjalan — hanya fitur riwayat yang dinonaktifkan.

```bash
python scripts/init_db.py
```

Skrip ini membuat database, tabel `predictions` dan `model_metrics`, lalu memuat
metrik dari `model_metrics.json` bila ada.

---

## Menjalankan Aplikasi

### Mode pengembangan

```bash
python app.py
```

Aplikasi berjalan di `http://127.0.0.1:5000`.

### Mode produksi (Gunicorn)

```bash
gunicorn --bind 0.0.0.0:8000 wsgi:app
```

---

## Deskripsi Halaman

| Halaman | Rute | Keterangan |
|---|---|---|
| Home | `/` | Ringkasan sistem + pemindai URL cepat |
| Deteksi URL | `/detect` | Form analisis URL dengan pilihan model |
| Hasil Analisis | `/result/<id>` | Verdict, probabilitas, risiko, SHAP, fitur |
| Perbandingan Algoritma | `/comparison` | Tabel & grafik metrik keempat model |
| Riwayat Prediksi | `/history` | Daftar prediksi tersimpan (butuh database) |
| Tentang | `/about` | Metodologi, algoritma, fitur, dan SHAP |

Endpoint API: `POST /api/detect` menerima JSON `{ "url": "...", "model": "..." }`
dan mengembalikan hasil prediksi beserta ID rekaman.

---

## Catatan Arsitektur

- Ekstraksi fitur bersifat **leksikal/struktural murni** — tidak ada permintaan jaringan,
  sehingga cepat dan aman (situs target tidak pernah dikunjungi).
- Model, scaler, dan explainer dimuat sekali ke memori (**singleton cache**) saat aplikasi
  start, sehingga prediksi berikutnya cepat.
- Kegagalan database **tidak pernah** menggagalkan proses prediksi; penyimpanan riwayat
  bersifat opsional.
- Model default dapat diubah lewat `DEFAULT_MODEL` di `config.py`.

---

## Alur Singkat Sebuah Prediksi

1. Input URL divalidasi & dibersihkan (`utils/validators.py`).
2. URL diubah menjadi 30 fitur numerik (`ml/feature_extractor.py`).
3. Fitur dinormalisasi dengan scaler, lalu diprediksi model terpilih.
4. SHAP menghitung kontribusi tiap fitur terhadap keputusan.
5. Probabilitas dipetakan ke tingkat risiko + saran tindakan.
6. Hasil ditampilkan dan (bila DB aktif) disimpan ke riwayat.

---

## Bobot Model Hasil Pelatihan (Pre-trained Models)

Untuk menjaga repository tetap ringan dan sesuai batasan ukuran file GitHub (<100MB), seluruh artefak model hasil training (`.pkl` dan `.json`) dikemas dalam arsip `models_store.zip` dan dapat diunduh melalui **[GitHub Releases](https://github.com/aryodwiprasetyo19-sys/phishing-detection/releases)**.

Setelah diunduh, ekstrak berkas tersebut ke folder `models_store/` sebelum menjalankan aplikasi.

---

## 🔒 Lisensi & Hak Cipta
**Copyright © 2026 Aryo Dwi Prasetyo — All Rights Reserved.**  
Repositori ini dipublikasikan semata-mata untuk tujuan demonstrasi portofolio dan evaluasi rekrutmen. Dilarang menyalin atau menggunakan ulang kode tanpa izin tertulis dari pemilik. Lihat berkas [LICENSE](./LICENSE) untuk detail lengkap.
