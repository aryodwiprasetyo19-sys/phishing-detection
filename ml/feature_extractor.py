# Copyright (c) 2026 Aryo Dwi Prasetyo - Universitas Gunadarma. All Rights Reserved.
"""
ml/feature_extractor.py
========================
Ekstraksi fitur dari URL mentah menjadi vektor numerik.

PENTING: Modul ini dipakai DUA KALI dan HARUS konsisten:
  1. Saat training  -> mengubah dataset URL menjadi matriks fitur.
  2. Saat inference  -> mengubah 1 URL input pengguna menjadi 1 baris fitur.

Bila urutan/definisi fitur berbeda antara training dan inference, prediksi akan
salah total. Karena itu daftar fitur (FEATURE_NAMES) dijadikan satu sumber
kebenaran dan metode extract() selalu mengembalikan fitur dalam urutan tersebut.

Semua fitur bersifat LEXICAL / STRUCTURAL (dihitung murni dari string URL),
tanpa koneksi internet (tanpa WHOIS/DNS/SSL). Ini disengaja agar:
  - Cepat dan deterministik (cocok untuk demo sidang).
  - Tidak bergantung pada ketersediaan jaringan atau layanan eksternal.
"""

import math
import re
from urllib.parse import urlparse

# Daftar TLD yang secara empiris sering disalahgunakan untuk phishing.
_SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "info", "club",
    "work", "click", "link", "loan", "download", "kim", "country",
    "science", "party", "gdn", "review", "stream", "bid", "date",
}

# Layanan pemendek URL yang umum (phishing sering menyembunyikan URL asli).
_SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "cutt.ly", "shorte.st", "rebrand.ly",
    "tiny.cc", "rb.gy", "shorturl.at",
}

# Nama brand yang paling sering ditiru pada serangan phishing (token lowercase,
# tanpa TLD). Deteksi berbasis substring pada string URL.
_BRANDS = {
    "paypal", "google", "apple", "appleid", "icloud", "microsoft", "outlook",
    "office365", "amazon", "aws", "facebook", "instagram", "whatsapp",
    "netflix", "ebay", "linkedin", "dropbox", "adobe", "coinbase", "binance",
    "metamask", "blockchain", "steam", "spotify", "twitter", "yahoo", "gmail",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc", "santander",
    "barclays", "americanexpress", "alibaba", "aliexpress", "wechat", "dhl",
    "fedex", "usps", "telegram", "tiktok", "roblox", "discord", "nintendo",
    "samsung", "walmart",
}

# Kelompok kata kunci "aksi" yang khas pada URL phishing. Setiap kelompok
# memetakan ke satu fitur biner contains_*_keyword. Substring dipilih hati-hati
# agar tidak salah cocok (mis. tidak memakai "pay" karena cocok dengan "paypal").
_KEYWORDS = {
    "login": ("login", "signin", "sign-in", "logon", "log-in", "authenticate"),
    "verify": ("verify", "verification", "verified", "validate", "validation"),
    "secure": ("secure", "security", "ssl"),
    "account": ("account", "acct"),
    "payment": ("payment", "billing", "invoice", "transaction", "checkout"),
    "update": ("update", "confirm", "renew", "suspend", "unlock", "reactivate", "recover"),
    "wallet": ("wallet", "crypto", "bitcoin", "blockchain", "ledger", "airdrop"),
}

# Sufiks TLD dua tingkat yang umum, untuk mengenali domain terdaftar dengan
# benar (mis. "co.uk", "co.id"). Tanpa ini, "example.co.uk" akan salah dibaca
# domain terdaftarnya sebagai "co.uk".
_MULTI_TLDS = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "co.id", "or.id", "ac.id", "go.id",
    "web.id", "com.au", "net.au", "org.au", "co.jp", "com.br", "com.cn",
    "com.sg", "com.my", "co.in", "co.za", "co.kr", "com.mx", "com.tr",
    "co.nz", "com.ph", "com.hk", "com.tw", "co.th", "com.vn",
}

# Karakter IPv4 dalam hostname (indikator kuat phishing).
_IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)$"
)


class URLFeatureExtractor:
    """Mengubah sebuah URL menjadi dictionary fitur numerik."""

    # Sumber kebenaran urutan fitur. WAJIB stabil antara training & inference.
    FEATURE_NAMES = [
        "url_length",
        "hostname_length",
        "path_length",
        "query_length",
        "count_dots",
        "count_hyphen",
        "count_underscore",
        "count_slash",
        "count_question",
        "count_equal",
        "count_at",
        "count_and",
        "count_percent",
        "count_digits",
        "count_letters",
        "count_params",
        "path_depth",
        "num_subdomains",
        "has_ip",
        "has_https",
        "has_port",
        "has_shortener",
        "has_suspicious_tld",
        "has_double_slash_in_path",
        "has_at_symbol",
        "http_in_path",
        "digit_letter_ratio",
        "special_char_ratio",
        "longest_word_length",
        "url_entropy",
        # --- Fitur berbasis brand & kata kunci (ditambahkan) ---
        "contains_brand_name",
        "contains_login_keyword",
        "contains_verify_keyword",
        "contains_secure_keyword",
        "contains_account_keyword",
        "contains_payment_keyword",
        "contains_update_keyword",
        "contains_wallet_keyword",
        "brand_in_subdomain",
        "brand_not_in_registered_domain",
    ]

    @staticmethod
    def _split_domain(hostname: str):
        """Pisahkan hostname menjadi (subdomain, domain_terdaftar).

        Domain terdaftar = domain + TLD (mis. "example.com"). Bila TLD dua
        tingkat (mis. "co.uk"), diambil tiga label terakhir. Mengembalikan
        ("", hostname) bila hostname tidak punya cukup label.
        """
        labels = [l for l in hostname.split(".") if l]
        if len(labels) < 2:
            return "", hostname
        last_two = ".".join(labels[-2:])
        if last_two in _MULTI_TLDS and len(labels) >= 3:
            registered = ".".join(labels[-3:])
            subdomain = ".".join(labels[:-3])
        else:
            registered = ".".join(labels[-2:])
            subdomain = ".".join(labels[:-2])
        return subdomain, registered

    @staticmethod
    def _parse_fallback(parse_target: str):
        """Parser cadangan (regex/split) untuk URL yang membuat urlparse()
        melempar ValueError, mis. karena ada "[...]" pada netloc/path yang
        bukan alamat IPv6 valid. Tidak melakukan validasi ketat seperti
        urlparse - cukup untuk keperluan ekstraksi fitur leksikal.
        """
        scheme = ""
        rest = parse_target
        sep = rest.find("://")
        if sep != -1:
            scheme = rest[:sep].lower()
            rest = rest[sep + 3:]

        rest = rest.split("#", 1)[0]
        query = ""
        if "?" in rest:
            rest, query = rest.split("?", 1)

        slash_idx = rest.find("/")
        if slash_idx == -1:
            netloc, path = rest, ""
        else:
            netloc, path = rest[:slash_idx], rest[slash_idx:]

        host_part = netloc.rsplit("@", 1)[-1]
        port = None
        if ":" in host_part:
            host_part, port_str = host_part.rsplit(":", 1)
            if port_str.isdigit():
                port = int(port_str)
        hostname = host_part.lower()

        return scheme, hostname, path, query, port

    def extract(self, url: str) -> dict:
        """Ekstrak seluruh fitur dari sebuah URL.

        Mengembalikan dict {nama_fitur: nilai} yang urutannya mengikuti
        FEATURE_NAMES. Input dinormalisasi terlebih dahulu.
        """
        url = (url or "").strip()

        # Tambahkan skema default jika pengguna hanya mengetik "contoh.com".
        # urlparse butuh skema agar netloc terisi dengan benar.
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
            parse_target = "http://" + url
        else:
            parse_target = url

        # Python 3.11+ memvalidasi apa pun di dalam "[...]" pada netloc sebagai
        # alamat IPv6 dan melempar ValueError bila bukan (lih. bpo-89970).
        # URL phishing/legit di dunia nyata sering memuat "[" "]" literal pada
        # query/path (mis. "?fields[]=x", "judul[Demo]"), jadi urlparse perlu
        # fallback manual agar ekstraksi tidak berhenti di tengah dataset.
        try:
            parsed = urlparse(parse_target)
            hostname = parsed.hostname or ""
            path = parsed.path or ""
            query = parsed.query or ""
            scheme = parsed.scheme
            port = parsed.port
        except ValueError:
            scheme, hostname, path, query, port = self._parse_fallback(parse_target)

        # Bagian setelah skema (tanpa "http://") untuk penghitungan karakter.
        url_body = parse_target.split("://", 1)[-1]

        features = {}

        # --- Panjang komponen ---
        features["url_length"] = len(url)
        features["hostname_length"] = len(hostname)
        features["path_length"] = len(path)
        features["query_length"] = len(query)

        # --- Frekuensi karakter khusus (dihitung pada bagian setelah skema) ---
        features["count_dots"] = url_body.count(".")
        features["count_hyphen"] = url_body.count("-")
        features["count_underscore"] = url_body.count("_")
        features["count_slash"] = url_body.count("/")
        features["count_question"] = url_body.count("?")
        features["count_equal"] = url_body.count("=")
        features["count_at"] = url_body.count("@")
        features["count_and"] = url_body.count("&")
        features["count_percent"] = url_body.count("%")

        digits = sum(c.isdigit() for c in url_body)
        letters = sum(c.isalpha() for c in url_body)
        features["count_digits"] = digits
        features["count_letters"] = letters

        # --- Struktur query & path ---
        features["count_params"] = len(query.split("&")) if query else 0
        features["path_depth"] = len([p for p in path.split("/") if p])

        # Jumlah subdomain: total label domain dikurangi domain+TLD (2 label).
        host_labels = [h for h in hostname.split(".") if h]
        features["num_subdomains"] = max(len(host_labels) - 2, 0)

        # --- Fitur biner (0/1) ---
        features["has_ip"] = 1 if _IP_PATTERN.match(hostname) else 0
        features["has_https"] = 1 if scheme == "https" else 0
        features["has_port"] = 1 if port is not None else 0
        features["has_shortener"] = 1 if hostname.lower() in _SHORTENERS else 0

        tld = host_labels[-1].lower() if host_labels else ""
        features["has_suspicious_tld"] = 1 if tld in _SUSPICIOUS_TLDS else 0

        features["has_double_slash_in_path"] = 1 if "//" in path else 0
        features["has_at_symbol"] = 1 if "@" in url_body else 0
        # "http" muncul lagi di path -> teknik penyamaran redirect.
        features["http_in_path"] = 1 if "http" in path.lower() else 0

        # --- Rasio ---
        features["digit_letter_ratio"] = round(digits / letters, 4) if letters else 0.0
        special = sum(not c.isalnum() for c in url_body)
        features["special_char_ratio"] = (
            round(special / len(url_body), 4) if url_body else 0.0
        )

        # --- Fitur lanjutan ---
        words = re.split(r"[^a-zA-Z0-9]+", url_body)
        features["longest_word_length"] = max((len(w) for w in words), default=0)
        features["url_entropy"] = self._shannon_entropy(url_body)

        # --- Fitur berbasis brand & kata kunci (leksikal) ---
        url_lower = url_body.lower()
        subdomain, registered = self._split_domain(hostname)
        sub_lower = subdomain.lower()
        reg_lower = registered.lower()

        # Brand mana saja yang benar-benar cocok (bukan sekadar "ada brand
        # apa pun"), supaya pengecekan domain terdaftar di bawah membandingkan
        # brand yang SAMA -- bukan brand berbeda yang kebetulan sama-sama ada
        # di URL (mis. platform hosting sah "sites.google.com" dipakai untuk
        # menyamar sebagai "wellsfargo" di path; "google" tidak boleh membuat
        # penyamaran "wellsfargo" itu lolos sebagai brand_not_in_registered_domain=0).
        matched_brands = {b for b in _BRANDS if b in url_lower}
        features["contains_brand_name"] = 1 if matched_brands else 0

        for key, tokens in _KEYWORDS.items():
            hit = any(tok in url_lower for tok in tokens)
            features[f"contains_{key}_keyword"] = 1 if hit else 0

        features["brand_in_subdomain"] = (
            1 if any(b in sub_lower for b in matched_brands) else 0
        )
        # Brand muncul di URL tetapi brand YANG SAMA itu tidak ada pada domain
        # terdaftar -> indikasi kuat penyamaran (mis. "paypal.com.aman-login.tk",
        # domain aslinya "aman-login.tk"; atau "sites.google.com/.../wellsfargologin",
        # domain aslinya milik Google, bukan Wells Fargo).
        features["brand_not_in_registered_domain"] = (
            1 if any(b not in reg_lower for b in matched_brands) else 0
        )

        return features

    def extract_ordered(self, url: str) -> list:
        """Ekstrak fitur sebagai list nilai sesuai urutan FEATURE_NAMES.

        Berguna untuk membentuk baris matriks fitur ML.
        """
        feats = self.extract(url)
        return [feats[name] for name in self.FEATURE_NAMES]

    @staticmethod
    def _shannon_entropy(text: str) -> float:
        """Menghitung entropi Shannon dari sebuah string.

        Entropi tinggi menandakan string acak (mis. domain yang di-generate
        otomatis), yang sering dikaitkan dengan URL phishing.
        """
        if not text:
            return 0.0
        length = len(text)
        counts = {}
        for ch in text:
            counts[ch] = counts.get(ch, 0) + 1
        entropy = 0.0
        for freq in counts.values():
            p = freq / length
            entropy -= p * math.log2(p)
        return round(entropy, 4)
