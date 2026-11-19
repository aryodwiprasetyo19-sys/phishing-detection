# Copyright (c) 2026 Aryo Dwi Prasetyo - Universitas Gunadarma. All Rights Reserved.
"""
utils/validators.py
===================
Validasi dan sanitasi input URL sebelum diproses oleh pipeline ML.

Tujuan: menolak input kosong/terlalu panjang/berformat jelas salah lebih awal,
sehingga pesan kesalahan yang jelas bisa diberikan ke pengguna dan pipeline ML
hanya menerima input yang wajar.
"""

import re

from config import Config

# Pola dasar: minimal ada "sesuatu.sesuatu" (host dengan titik) atau alamat IP,
# dengan atau tanpa skema http/https di depan.
_URL_PATTERN = re.compile(
    r"^(https?://)?"                 # skema opsional
    r"([\w\-]+(\.[\w\-]+)+)"          # host: minimal satu titik (domain.tld)
    r"(:\d+)?"                        # port opsional
    r"(/[^\s]*)?$",                   # path/query opsional (tanpa spasi)
    re.IGNORECASE,
)

# Pola alamat IP (dengan skema & path opsional).
_IP_URL_PATTERN = re.compile(
    r"^(https?://)?"
    r"(\d{1,3}\.){3}\d{1,3}"
    r"(:\d+)?"
    r"(/[^\s]*)?$",
)


def validate_url(url: str):
    """Validasi URL. Return tuple (is_valid: bool, message: str | cleaned_url).

    Bila valid, elemen kedua berisi URL yang sudah dirapikan.
    Bila tidak valid, elemen kedua berisi pesan kesalahan untuk pengguna.
    """
    if url is None:
        return False, "URL tidak boleh kosong."

    cleaned = url.strip()
    if not cleaned:
        return False, "URL tidak boleh kosong."

    if len(cleaned) > Config.MAX_URL_LENGTH:
        return False, f"URL terlalu panjang (maksimal {Config.MAX_URL_LENGTH} karakter)."

    if " " in cleaned:
        return False, "URL tidak boleh mengandung spasi."

    if _URL_PATTERN.match(cleaned) or _IP_URL_PATTERN.match(cleaned):
        return True, cleaned

    return False, "Format URL tidak valid. Contoh yang benar: https://contoh.com/halaman"
