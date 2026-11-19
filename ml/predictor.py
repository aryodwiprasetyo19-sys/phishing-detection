# Copyright (c) 2026 Aryo Dwi Prasetyo - Universitas Gunadarma. All Rights Reserved.
"""
ml/predictor.py
===============.

Fungsi utama predict(url, model_name) menjalankan alur end-to-end:
    URL
      -> ekstraksi 40 fitur (feature_extractor)
      -> scaling (scaler bersama)
      -> prediksi probabilitas phishing (model terpilih)
      -> penjelasan SHAP per fitur (explainer)
      -> penentuan risk level (Low/Medium/High)
      -> pembuatan saran otomatis berbasis fitur mencurigakan dominan

Hasilnya berupa dict lengkap yang siap disimpan ke DB dan dirender ke halaman
hasil analisis.
"""

import numpy as np
from urllib.parse import urlparse

from config import (
    CLASSIFICATION_THRESHOLD,
    DEFAULT_MODEL,
    MODEL_NAMES,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_LOW,
)
from ml import model_loader
from ml.feature_extractor import URLFeatureExtractor
from utils import whois_lookup

_extractor = URLFeatureExtractor()

# Kunci & label khusus untuk hasil gabungan (ensemble) keempat algoritma.
ENSEMBLE_KEY = "ensemble"
ENSEMBLE_DISPLAY = "Gabungan 4 Algoritma"

# Domain resmi brand besar yang sudah diverifikasi manual (cocok dengan katalog
# _BRANDS di feature_extractor.py). Fitur leksikal murni tidak bisa membedakan
# domain resmi varian brand (mis. "microsoftonline.com") dari domain palsu yang
# menyisipkan nama brand (mis. "paypal-securecheck-update.com") -- keduanya
# sama-sama mengandung brand sebagai substring. Whitelist ini menjadi jalan
# pintas yang pasti benar untuk brand-brand besar yang paling sering ditiru.
#
# SENGAJA TIDAK memasukkan domain hosting multi-tenant (mis. google.com,
# googleusercontent.com, dropbox.com) walau brand itu ada di katalog _BRANDS:
# domain semacam itu mengizinkan siapa pun mengunggah konten pihak ketiga
# (Google Sites/Forms/Docs, berkas Dropbox yang dibagikan), sehingga sering
# disalahgunakan justru UNTUK menghosting halaman phishing brand lain
# (mis. "sites.google.com/view/wellsfargologin"). Meng-whitelist domain
# tersebut akan membuat teknik penyamaran ini lolos begitu saja.
_TRUSTED_DOMAINS = {
    "microsoft.com", "microsoftonline.com", "live.com", "office.com", "office365.com",
    "youtube.com", "gstatic.com",
    "apple.com", "icloud.com",
    "amazon.com", "amazon.co.id",
    "paypal.com",
    "facebook.com", "instagram.com", "whatsapp.com",
    "netflix.com",
    "ebay.com",
    "linkedin.com",
    "adobe.com",
    "coinbase.com",
    "binance.com",
    "spotify.com",
    "twitter.com", "x.com",
    "yahoo.com", "gmail.com",
    "chase.com", "wellsfargo.com", "bankofamerica.com", "citibank.com",
    "hsbc.com", "santander.com", "barclays.co.uk", "americanexpress.com",
    "alibaba.com", "aliexpress.com",
    "dhl.com", "fedex.com", "usps.com",
    "telegram.org",
    "tiktok.com",
    "roblox.com",
    "discord.com",
    "nintendo.com",
    "samsung.com",
    "walmart.com",
}

# Subdomain resmi otentikasi/akun tertentu yang diverifikasi manual, pada
# domain INDUK yang SENGAJA tidak masuk _TRUSTED_DOMAINS karena bersifat
# hosting multi-tenant (mis. google.com -- lihat komentar di atas). Berbeda
# dari _TRUSTED_DOMAINS yang mencocokkan domain terdaftar (registered
# domain), daftar ini mencocokkan HOSTNAME LENGKAP secara persis, sehingga
# hanya subdomain resmi ini yang diredam -- subdomain hosting lain di domain
# induk yang sama (mis. sites.google.com, forms.google.com) TETAP tidak
# dipercaya dan tetap bisa terdeteksi sebagai phishing.
_TRUSTED_SUBDOMAINS = {
    "accounts.google.com",
    "myaccount.google.com",
}


def _trust_adjustment(hostname: str, registered_domain: str, features: dict, age_days) -> float:
    """Faktor pengali (<= 1.0) untuk meredam skor phishing berdasarkan sinyal
    kepercayaan domain yang tidak bisa ditangkap fitur leksikal saja.

    Hanya MEREDAM (tidak pernah menaikkan) skor, dan tidak diterapkan bila ada
    sinyal bahaya keras (IP, TLD mencurigakan, shortener) supaya domain tua
    yang disalahgunakan tetap terdeteksi.
    """
    # Ada indikasi brand lain ditiru (brand muncul di URL tapi bukan pada
    # domain terdaftarnya) -> jangan pernah diredam, apa pun domainnya/umurnya.
    if features.get("brand_not_in_registered_domain"):
        return 1.0

    if hostname in _TRUSTED_SUBDOMAINS or registered_domain in _TRUSTED_DOMAINS:
        return 0.05

    has_hard_risk = (
        features.get("has_ip")
        or features.get("has_suspicious_tld")
        or features.get("has_shortener")
    )
    if age_days is not None and not has_hard_risk:
        if age_days >= 3650:  # >= 10 tahun
            return 0.5
        if age_days >= 1825:  # >= 5 tahun
            return 0.75
    return 1.0

# Label fitur berbahasa Indonesia yang mudah dipahami awam.
# Disamakan dengan FEATURE_LABELS di static/js/shap_chart.js.
FEATURE_LABELS_ID = {
    "url_length": "Panjang URL",
    "hostname_length": "Panjang nama host",
    "path_length": "Panjang path",
    "query_length": "Panjang query",
    "count_dots": "Jumlah titik",
    "count_hyphen": "Jumlah tanda hubung",
    "count_underscore": "Jumlah garis bawah",
    "count_slash": "Jumlah garis miring",
    "count_question": "Jumlah tanda '?'",
    "count_equal": "Jumlah tanda '='",
    "count_at": "Jumlah simbol '@'",
    "count_and": "Jumlah tanda '&'",
    "count_percent": "Jumlah tanda '%'",
    "count_digits": "Jumlah digit",
    "count_letters": "Jumlah huruf",
    "count_params": "Jumlah parameter",
    "path_depth": "Kedalaman path",
    "num_subdomains": "Jumlah subdomain",
    "has_ip": "Penggunaan alamat IP",
    "has_https": "Penggunaan HTTPS",
    "has_port": "Penggunaan port",
    "has_shortener": "URL shortener",
    "has_suspicious_tld": "TLD mencurigakan",
    "has_double_slash_in_path": "Double slash di path",
    "has_at_symbol": "Simbol '@'",
    "http_in_path": "'http' di dalam path",
    "digit_letter_ratio": "Rasio digit/huruf",
    "special_char_ratio": "Rasio karakter khusus",
    "longest_word_length": "Panjang kata terpanjang",
    "url_entropy": "Keacakan karakter (entropi)",
    "contains_brand_name": "Mengandung nama brand",
    "contains_login_keyword": "Kata kunci 'login'",
    "contains_verify_keyword": "Kata kunci 'verify'",
    "contains_secure_keyword": "Kata kunci 'secure'",
    "contains_account_keyword": "Kata kunci 'account'",
    "contains_payment_keyword": "Kata kunci pembayaran",
    "contains_update_keyword": "Kata kunci 'update'",
    "contains_wallet_keyword": "Kata kunci dompet/kripto",
    "brand_in_subdomain": "Nama brand di subdomain",
    "brand_not_in_registered_domain": "Brand di luar domain terdaftar",
}


def _reason_phrase(name: str, value) -> str:
    """Susun kalimat interpretatif ramah-awam untuk sebuah fitur & nilainya.

    Fitur biner (has_*) menghasilkan kalimat ya/tidak; fitur jumlah/panjang
    menyertakan nilainya agar pengguna paham dari mana penilaian berasal.
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = 0.0
    iv = int(round(num))

    templates = {
        "url_length": f"Panjang URL {iv} karakter",
        "hostname_length": f"Panjang nama host {iv} karakter",
        "path_length": f"Panjang path {iv} karakter",
        "query_length": f"Panjang query {iv} karakter",
        "count_dots": f"Terdapat {iv} tanda titik pada URL",
        "count_hyphen": f"Terdapat {iv} tanda hubung '-'",
        "count_underscore": f"Terdapat {iv} garis bawah '_'",
        "count_slash": f"Terdapat {iv} garis miring '/'",
        "count_question": f"Terdapat {iv} tanda '?'",
        "count_equal": f"Terdapat {iv} tanda '='",
        "count_at": f"Terdapat {iv} simbol '@'",
        "count_and": f"Terdapat {iv} tanda '&'",
        "count_percent": f"Terdapat {iv} tanda '%'",
        "count_digits": f"Terdapat {iv} digit angka",
        "count_letters": f"Terdapat {iv} huruf",
        "count_params": f"Terdapat {iv} parameter query",
        "path_depth": f"Kedalaman path {iv} tingkat",
        "num_subdomains": f"Memiliki {iv} subdomain",
        "has_ip": "Menggunakan alamat IP sebagai host (bukan nama domain)"
        if num >= 1 else "Tidak menggunakan alamat IP",
        "has_https": "Menggunakan HTTPS"
        if num >= 1 else "Tidak menggunakan HTTPS (koneksi tidak terenkripsi)",
        "has_port": "Menyertakan nomor port pada URL"
        if num >= 1 else "Tanpa nomor port khusus",
        "has_shortener": "Menggunakan layanan pemendek URL"
        if num >= 1 else "Bukan URL yang dipendekkan",
        "has_suspicious_tld": "Memakai ekstensi domain (TLD) yang sering dipakai phishing"
        if num >= 1 else "Ekstensi domain (TLD) tergolong umum",
        "has_double_slash_in_path": "Terdapat '//' ganda di dalam path"
        if num >= 1 else "Tidak ada '//' ganda pada path",
        "has_at_symbol": "URL mengandung simbol '@'"
        if num >= 1 else "URL tidak mengandung simbol '@'",
        "http_in_path": "Kata 'http' muncul lagi di dalam path"
        if num >= 1 else "Tidak ada 'http' tersembunyi di path",
        "digit_letter_ratio": f"Rasio angka terhadap huruf {num:.2f}",
        "special_char_ratio": f"Rasio karakter khusus {num:.2f}",
        "longest_word_length": f"Kata terpanjang {iv} karakter",
        "url_entropy": f"Tingkat keacakan karakter {num:.2f}",
        "contains_brand_name": "URL mencantumkan nama merek terkenal"
        if num >= 1 else "Tidak mencantumkan nama merek terkenal",
        "contains_login_keyword": "Mengandung kata 'login/sign-in'"
        if num >= 1 else "Tanpa kata 'login'",
        "contains_verify_keyword": "Mengandung kata 'verify/verifikasi'"
        if num >= 1 else "Tanpa kata 'verify'",
        "contains_secure_keyword": "Mengandung kata 'secure/security'"
        if num >= 1 else "Tanpa kata 'secure'",
        "contains_account_keyword": "Mengandung kata 'account/akun'"
        if num >= 1 else "Tanpa kata 'account'",
        "contains_payment_keyword": "Mengandung kata terkait pembayaran (payment/billing)"
        if num >= 1 else "Tanpa kata terkait pembayaran",
        "contains_update_keyword": "Mengandung kata desakan (update/confirm/verifikasi ulang)"
        if num >= 1 else "Tanpa kata desakan",
        "contains_wallet_keyword": "Mengandung kata dompet/kripto (wallet/crypto)"
        if num >= 1 else "Tanpa kata dompet/kripto",
        "brand_in_subdomain": "Nama merek diletakkan di subdomain (teknik penyamaran)"
        if num >= 1 else "Tidak ada merek di subdomain",
        "brand_not_in_registered_domain": "Nama merek muncul, tetapi domain resminya bukan milik merek itu (indikasi kuat penyamaran)"
        if num >= 1 else "Merek (jika ada) sesuai dengan domain resminya",
    }
    return templates.get(name, FEATURE_LABELS_ID.get(name, name))


def build_explanation_reasons(top_features, features, signed=True, limit=6):
    """Ubah fitur paling berpengaruh menjadi daftar alasan berbahasa manusia.

    Parameters
    ----------
    top_features : list[tuple(nama, nilai_shap)]
        Fitur terurut menurut pengaruh (|SHAP|) menurun.
    features : dict
        Nilai fitur mentah URL (untuk menyusun kalimat + menampilkan nilainya).
    signed : bool
        True bila nilai berasal dari SHAP (bertanda -> punya arah phishing/aman).
        False bila dari feature_importances_ (tak bertanda -> arah 'netral').
    limit : int
        Jumlah alasan maksimum yang ditampilkan.

    Returns
    -------
    list[dict] : {label, text, direction, impact}
        direction: 'phishing' (menaikkan risiko), 'aman' (menurunkan risiko),
        atau 'netral' (hanya berpengaruh, tanpa arah).
    """
    feats = features or {}
    reasons = []
    for name, shap_val in (top_features or [])[:limit]:
        try:
            sv = float(shap_val)
        except (TypeError, ValueError):
            sv = 0.0
        if not signed:
            direction = "netral"
        else:
            direction = "phishing" if sv > 0 else "aman"
        reasons.append(
            {
                "label": FEATURE_LABELS_ID.get(name, name),
                "text": _reason_phrase(name, feats.get(name, 0)),
                "direction": direction,
                "impact": round(abs(sv), 6),
            }
        )
    return reasons


def _short_explanation(top_features, features, signed=True, limit=2) -> str:
    """Ringkasan 1 kalimat kenapa satu algoritma memberi skor tertentu.

    Dipakai pada breakdown "Perbandingan 4 Algoritma" di halaman hasil agar
    setiap algoritma punya penjelasan singkat sendiri (bukan cuma angka
    probabilitas telanjang).
    """
    reasons = build_explanation_reasons(top_features, features, signed=signed, limit=limit)
    if not reasons:
        return "Tidak ada fitur dominan yang teridentifikasi untuk algoritma ini."
    return "Faktor paling berpengaruh: " + "; ".join(r["text"] for r in reasons) + "."


def _hostname_of(url: str) -> str:
    """Ambil hostname dari URL (menambahkan skema bila perlu)."""
    target = url if "://" in url else "http://" + url
    return (urlparse(target).hostname or "").lower()


def build_narrative(features, label, risk_level, domain_age_info=None, shap_dict=None):
    """Susun 'Analisis AI' berbahasa awam dari sinyal yang benar-benar dimiliki.

    Hanya memakai sinyal dari URL + WHOIS (tanpa mengunjungi/menganalisis isi
    halaman). Mengembalikan dict {points: [{text, direction}], conclusion}.

    Bila `shap_dict` (nilai SHAP bertanda per fitur) tersedia, arah tiap poin
    ("menaikkan"/"menurunkan indikasi phishing") mengikuti tanda SHAP fitur
    terkait, bukan asumsi tetap. Ini membuat "Analisis AI" selalu sinkron
    dengan bagian "Explainable AI (SHAP)" di bawahnya, yang memakai sumber
    data yang sama. Kesimpulan juga mengikuti `label` (bukan hanya
    `risk_level`) agar tidak bertentangan dengan status/verdict yang
    ditampilkan di atas.
    """
    f = features
    points = []
    has_signed_shap = bool(shap_dict)

    def shap_direction(feature_key, default):
        if not has_signed_shap or feature_key is None:
            return default
        sv = shap_dict.get(feature_key)
        if sv is None:
            return default
        return "phishing" if sv > 0 else "aman"

    def combined_shap_direction(feature_keys, default):
        if not has_signed_shap:
            return default
        candidates = [(k, shap_dict[k]) for k in feature_keys if k in shap_dict]
        if not candidates:
            return default
        # Fitur dengan pengaruh (|SHAP|) terbesar menentukan arah gabungan.
        _, sv = max(candidates, key=lambda kv: abs(kv[1]))
        return "phishing" if sv > 0 else "aman"

    def add(cond, text, feature_key=None, direction="phishing"):
        if cond:
            points.append({"text": text, "direction": shap_direction(feature_key, direction)})

    # --- Sinyal penyamaran merek (paling kuat) ---
    add(
        f.get("brand_not_in_registered_domain"),
        "URL menampilkan nama merek terkenal, padahal domain resminya bukan milik "
        "merek tersebut - ini ciri khas upaya penyamaran (phishing).",
        feature_key="brand_not_in_registered_domain",
    )
    add(
        f.get("brand_in_subdomain") and not f.get("brand_not_in_registered_domain"),
        "Nama merek diletakkan pada bagian subdomain agar alamat tampak resmi.",
        feature_key="brand_in_subdomain",
    )

    # --- Sinyal struktur/host mencurigakan ---
    add(
        f.get("has_suspicious_tld"),
        "Domain memakai ekstensi (seperti .xyz, .tk, atau .top) yang sering dipakai "
        "untuk phishing karena murah dan mudah didaftarkan.",
        feature_key="has_suspicious_tld",
    )
    add(
        f.get("has_ip"),
        "Alamat situs menggunakan angka IP, bukan nama domain yang wajar - sering "
        "dipakai penipu untuk menyembunyikan identitas situs.",
        feature_key="has_ip",
    )
    add(
        not f.get("has_https"),
        "Situs tidak menggunakan HTTPS, sehingga data yang dikirim tidak terenkripsi "
        "dan rentan disadap.",
        feature_key="has_https",
    )
    add(
        f.get("has_https"),
        "Situs menggunakan HTTPS, sehingga data yang dikirim ke server terenkripsi.",
        feature_key="has_https",
        direction="aman",
    )
    add(
        f.get("has_shortener"),
        "Alamat asli disembunyikan di balik layanan pemendek URL.",
        feature_key="has_shortener",
    )
    add(
        f.get("has_at_symbol"),
        "URL mengandung simbol '@' yang dapat menyamarkan alamat tujuan sebenarnya.",
        feature_key="has_at_symbol",
    )

    # --- Kata kunci mendesak khas penipuan ---
    keyword_map = {
        "contains_login_keyword": "login",
        "contains_verify_keyword": "verifikasi",
        "contains_secure_keyword": "secure",
        "contains_account_keyword": "akun",
        "contains_payment_keyword": "pembayaran",
        "contains_update_keyword": "update/konfirmasi",
        "contains_wallet_keyword": "dompet/kripto",
    }
    hit_keys = [k for k in keyword_map if f.get(k)]
    if hit_keys:
        points.append(
            {
                "text": "URL memuat kata-kata yang umum pada penipuan (" +
                ", ".join(keyword_map[k] for k in hit_keys) +
                ") untuk mendorong korban buru-buru memasukkan data.",
                "direction": combined_shap_direction(hit_keys, "phishing"),
            }
        )

    # --- Umur domain (WHOIS, bila tersedia) ---
    if domain_age_info and domain_age_info.get("available"):
        extra = (
            " Situs phishing biasanya berumur sangat muda."
            if domain_age_info["direction"] == "phishing" else ""
        )
        points.append(
            {
                "text": domain_age_info["text"] + "." + extra,
                "direction": domain_age_info["direction"],
            }
        )

    # --- Bila nyaris tak ada tanda bahaya ---
    if not points:
        points.append(
            {
                "text": "Tidak ditemukan ciri phishing yang menonjol pada alamat ini "
                "(tanpa penyamaran merek, tanpa ekstensi mencurigakan, struktur wajar).",
                "direction": "aman",
            }
        )

    # --- Kalimat kesimpulan mengikuti label akhir + tingkat risiko, agar
    # tidak pernah berlawanan dengan badge status yang ditampilkan di atas ---
    if label == "phishing":
        if risk_level == "High":
            conclusion = (
                "Situs ini sangat mencurigakan dan memiliki indikasi kuat sebagai upaya "
                "phishing. Jangan memasukkan data pribadi, kata sandi, atau informasi "
                "kartu di halaman ini."
            )
        else:
            conclusion = (
                "Sistem mengklasifikasikan situs ini sebagai berpotensi phishing "
                "berdasarkan gabungan faktor di atas. Berhati-hatilah dan pastikan "
                "Anda mengakses situs resmi sebelum memasukkan data apa pun."
            )
    else:
        if risk_level == "Low":
            conclusion = (
                "Tidak ditemukan indikasi phishing yang kuat. Meski begitu, tetap periksa "
                "ejaan alamat dan pastikan situsnya resmi sebelum memasukkan data pribadi."
            )
        else:
            conclusion = (
                "Situs ini tergolong kemungkinan aman, namun tetap memiliki beberapa "
                "tanda yang perlu diwaspadai. Pastikan Anda mengakses situs resmi "
                "sebelum memasukkan data apa pun."
            )

    return {"points": points, "conclusion": conclusion}


def _determine_risk_level(prob: float) -> str:
    """Petakan probabilitas phishing ke tingkat risiko."""
    if prob < RISK_THRESHOLD_LOW:
        return "Low"
    if prob < RISK_THRESHOLD_HIGH:
        return "Medium"
    return "High"


def _shap_to_dict(shap_row, feature_names) -> dict:
    """Konversi satu baris nilai SHAP menjadi {fitur: kontribusi}.

    SHAP mengembalikan bentuk berbeda tergantung model/versi. Fungsi ini
    menormalkan semuanya menjadi vektor 1D (kontribusi kelas phishing untuk
    1 sampel), lalu memetakannya ke nama fitur. Bentuk yang ditangani:
      - list [arr_kelas0, arr_kelas1]     -> np.asarray jadi (2, 1, n)
      - array (1, n)                        -> (n_samples, n_features)
      - array (2, n)                        -> (n_classes, n_features)
      - array (1, n, 2)                     -> (n_samples, n_features, n_classes)
      - shap.Explanation (punya .values)
    """
    values = shap_row
    if hasattr(values, "values"):  # objek shap.Explanation
        values = values.values

    values = np.asarray(values, dtype=float)
    n = len(feature_names)

    if values.ndim == 1:
        vec = values
    elif values.ndim == 2:
        # (n_classes, n_features) untuk klasifikasi biner -> ambil kelas 1
        if values.shape == (2, n):
            vec = values[1]
        else:  # (n_samples, n_features) -> ambil sampel pertama
            vec = values[0]
    elif values.ndim == 3:
        # (n_classes, n_samples, n_features): kelas terakhir, sampel pertama
        if values.shape[0] <= 2 and values.shape[-1] == n:
            vec = values[-1, 0, :]
        # (n_samples, n_features, n_classes): sampel pertama, kelas terakhir
        elif values.shape[1] == n:
            vec = values[0, :, -1]
        else:
            vec = values.reshape(-1)[-n:]
    else:
        vec = values.reshape(-1)[:n]

    vec = np.asarray(vec).reshape(-1)
    return {name: round(float(v), 6) for name, v in zip(feature_names, vec)}


def _build_suggestion(label: str, risk_level: str, explanation_reasons: list) -> str:
    """Susun saran otomatis berbasis alasan yang benar-benar menaikkan risiko.

    Memakai `explanation_reasons` (sudah value-aware, sama seperti yang
    ditampilkan di bagian SHAP) dan hanya mengambil yang arahnya "phishing",
    agar kalimat saran tidak pernah bertentangan dengan nilai fitur asli.
    """
    if label == "legitimate" and risk_level == "Low":
        return (
            "URL ini terindikasi aman berdasarkan analisis fitur. Meski demikian, "
            "tetap periksa kembali ejaan domain dan pastikan Anda mengakses situs "
            "resmi sebelum memasukkan data pribadi."
        )

    reasons = [
        r["text"] for r in explanation_reasons if r.get("direction") == "phishing"
    ][:3]

    intro = {
        "High": "URL ini berisiko TINGGI sebagai phishing. Sangat disarankan untuk TIDAK mengaksesnya.",
        "Medium": "URL ini memiliki risiko SEDANG. Berhati-hatilah sebelum mengaksesnya.",
        "Low": "URL ini memiliki beberapa indikasi mencurigakan meski risikonya rendah.",
    }[risk_level]

    if reasons:
        detail = " Alasan utama: " + ". ".join(reasons) + "."
    else:
        detail = (
            " Beberapa karakteristik URL menyimpang dari pola situs tepercaya. "
            "Jangan memasukkan kredensial atau data keuangan pada halaman ini."
        )
    return intro + detail


def _extract_and_scale(url: str):
    """Ekstraksi fitur + scaling. Kembalikan (features, X, feature_names)."""
    feature_names = _extractor.FEATURE_NAMES
    features = _extractor.extract(url)
    vector = np.array([[features[n] for n in feature_names]], dtype=float)
    # Bersihkan nilai tak wajar (NaN/inf) agar model tidak error.
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = model_loader.get_scaler()
    X = scaler.transform(vector) if scaler is not None else vector
    return features, X, feature_names


def _score_model(model_name: str, X, feature_names):
    """Skor satu model. Kembalikan (phishing_prob, shap_dict)."""
    model = model_loader.get_model(model_name)
    phishing_prob = float(model.predict_proba(X)[0][1])
    shap_dict = {}
    explainer = model_loader.get_explainer(model_name)
    if explainer is not None:
        try:
            raw = explainer.shap_values(X)
            shap_dict = _shap_to_dict(raw, feature_names)
        except Exception:
            shap_dict = {}
    # Fallback importances (tak bertanda) bila SHAP tak ada.
    if not shap_dict and hasattr(model, "feature_importances_"):
        shap_dict = {
            n: round(float(v), 6)
            for n, v in zip(feature_names, model.feature_importances_)
        }
        shap_dict["__unsigned__"] = True  # penanda internal
    return phishing_prob, shap_dict


def _rank_top(shap_dict, limit=10):
    """Urutkan fitur menurut |kontribusi| menurun (abaikan penanda internal)."""
    items = [(k, v) for k, v in shap_dict.items() if k != "__unsigned__"]
    return sorted(items, key=lambda kv: abs(kv[1]), reverse=True)[:limit]


def _per_model_entries(X, feature_names, features, trust_factor):
    """Skor SEMUA algoritma yang tersedia secara terpisah pada X yang sama.

    Dipakai oleh score_all_models() untuk membangun breakdown "Perbandingan
    4 Algoritma" pada halaman hasil.

    Returns
    -------
    (entries, shap_accum, shap_count)
        entries     : list[dict] hasil per model (siap dirender).
        shap_accum  : dict {fitur: jumlah kontribusi SHAP lintas model}.
        shap_count  : jumlah model yang punya SHAP bertanda (untuk dirata-rata).
    """
    available = model_loader.available_models()
    entries = []
    shap_accum, shap_count = {}, 0
    for name in available:
        prob, shap_dict = _score_model(name, X, feature_names)
        prob = float(prob) * trust_factor
        signed = bool(shap_dict) and not shap_dict.get("__unsigned__")
        top_feats = _rank_top(shap_dict, limit=3)
        entries.append(
            {
                "model_used": name,
                "model_display": MODEL_NAMES.get(name, name),
                "phishing_probability": round(prob, 4),
                "phishing_probability_pct": round(prob * 100, 2),
                "prediction_label": (
                    "phishing" if prob >= CLASSIFICATION_THRESHOLD else "legitimate"
                ),
                "risk_level": _determine_risk_level(prob),
                "explanation": _short_explanation(top_feats, features, signed=signed),
            }
        )
        if shap_dict:
            for k, v in shap_dict.items():
                if k == "__unsigned__":
                    continue
                shap_accum[k] = shap_accum.get(k, 0.0) + v
            shap_count += 1
    return entries, shap_accum, shap_count


def score_all_models(url: str, features: dict, age_days) -> list:
    """Skor ulang sebuah URL dengan KEEMPAT algoritma secara terpisah, dari
    fitur yang sudah diekstrak sebelumnya (tanpa ekstraksi/WHOIS ulang).

    Dipakai halaman hasil (/result/<id>) agar hasil gabungan (ensemble) yang
    tersimpan tetap bisa ditampilkan berdampingan dengan pendapat
    masing-masing algoritma, bukan cuma hasil ensemble saja.
    """
    feature_names = _extractor.FEATURE_NAMES
    vector = np.array([[features.get(n, 0) for n in feature_names]], dtype=float)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = model_loader.get_scaler()
    X = scaler.transform(vector) if scaler is not None else vector

    hostname = _hostname_of(url)
    _, registered_domain = _extractor._split_domain(hostname)
    trust_factor = _trust_adjustment(hostname, registered_domain, features, age_days)

    entries, _, _ = _per_model_entries(X, feature_names, features, trust_factor)
    return entries


def predict(url: str, mode: str = ENSEMBLE_KEY) -> dict:
    """Deteksi phishing end-to-end untuk sebuah URL.

    Parameters
    ----------
    url : str
        URL yang akan dianalisis.
    mode : str
        - "ensemble" (default): gabungkan SEMUA algoritma menjadi satu hasil
          (rata-rata probabilitas & rata-rata nilai SHAP).
        - salah satu kunci di MODEL_NAMES: pakai satu algoritma saja.

    Returns
    -------
    dict lengkap hasil analisis (siap disimpan & dirender).
    """
    features, X, feature_names = _extract_and_scale(url)

    available = model_loader.available_models()
    if not available:
        raise RuntimeError(
            "Belum ada model yang dilatih. Jalankan notebook/pipeline training dulu."
        )

    if mode != ENSEMBLE_KEY and mode in available:
        # --- Mode satu algoritma ---
        prob, shap_dict = _score_model(mode, X, feature_names)
        model_used, model_display = mode, MODEL_NAMES[mode]
        used_shap = bool(shap_dict) and not shap_dict.get("__unsigned__")
    else:
        # --- Mode ensemble: rata-rata seluruh model yang tersedia ---
        probs, shap_accum, shap_count, any_signed = [], {}, 0, False
        for name in available:
            p, sd = _score_model(name, X, feature_names)
            probs.append(p)
            if sd:
                signed = not sd.get("__unsigned__")
                any_signed = any_signed or signed
                for k, v in sd.items():
                    if k == "__unsigned__":
                        continue
                    shap_accum[k] = shap_accum.get(k, 0.0) + v
                shap_count += 1
        prob = float(np.mean(probs)) if probs else 0.0
        shap_dict = (
            {k: round(v / shap_count, 6) for k, v in shap_accum.items()}
            if shap_count else {}
        )
        model_used, model_display = ENSEMBLE_KEY, ENSEMBLE_DISPLAY
        used_shap = any_signed

    # Sinyal tambahan: umur domain via WHOIS (situasional; None bila gagal/limit).
    hostname = _hostname_of(url)
    _, registered_domain = _extractor._split_domain(hostname)
    age_days = whois_lookup.get_domain_age_days(hostname)
    domain_age_info = whois_lookup.describe_domain_age(age_days)

    # Redam skor untuk domain resmi brand besar / domain yang sudah lama
    # terdaftar -- sinyal yang tak bisa ditangkap fitur leksikal saja.
    prob = float(prob) * _trust_adjustment(hostname, registered_domain, features, age_days)

    label = "phishing" if prob >= CLASSIFICATION_THRESHOLD else "legitimate"
    risk_level = _determine_risk_level(prob)
    top_features = _rank_top(shap_dict)

    explanation_reasons = build_explanation_reasons(
        top_features, features, signed=used_shap
    )
    clean_shap = {k: v for k, v in shap_dict.items() if k != "__unsigned__"}
    narrative = build_narrative(
        features, label, risk_level, domain_age_info,
        shap_dict=clean_shap if used_shap else None,
    )
    suggestion = _build_suggestion(label, risk_level, explanation_reasons)

    # Pendapat masing-masing algoritma -- hanya relevan pada mode gabungan
    # (ensemble); bila pengguna memilih satu algoritma saja, tidak ada
    # "algoritma lain" yang perlu dibandingkan.
    per_model = (
        score_all_models(url, features, age_days)
        if model_used == ENSEMBLE_KEY else []
    )

    return {
        "url": url,
        "mode": model_used,
        "model_used": model_used,
        "model_display": model_display,
        "prediction_label": label,
        "phishing_probability": round(prob, 4),
        "phishing_probability_pct": round(prob * 100, 2),
        "risk_level": risk_level,
        "extracted_features": features,
        "shap_values": clean_shap,
        "top_features": top_features,
        "explanation_reasons": explanation_reasons,
        "narrative": narrative,
        "domain_age_days": age_days,
        "suggestion": suggestion,
        "best_model": model_loader.best_model(),
        "per_model": per_model,
    }


