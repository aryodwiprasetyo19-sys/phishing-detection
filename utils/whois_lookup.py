# Copyright (c) 2026 Aryo Dwi Prasetyo - Universitas Gunadarma. All Rights Reserved.
"""
utils/whois_lookup.py
=====================
Pengambilan UMUR DOMAIN via WHOIS — bersifat OPSIONAL & SITUASIONAL.

Sesuai desain: umur domain BUKAN fitur yang dilatih (tidak masuk vektor fitur
model), melainkan sinyal tambahan real-time saat prediksi untuk memperkaya
penjelasan. Bila WHOIS gagal, timeout, atau kena rate-limit, fungsi cukup
mengembalikan None dan sistem tetap berjalan normal tanpa sinyal ini.

Dependensi: python-whois (lihat requirements.txt). Bila paket tidak terpasang,
modul tetap aman diimpor dan get_domain_age_days() mengembalikan None.
"""

import re
import socket
from datetime import datetime

try:  # Impor defensif: app tidak boleh gagal hanya karena WHOIS tak terpasang.
    import whois as _whois
except Exception:  # pragma: no cover - tergantung lingkungan
    _whois = None

# Pola IPv4 sederhana; host berupa IP tidak punya WHOIS domain yang bermakna.
_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def is_available() -> bool:
    """True bila pustaka WHOIS terpasang dan dapat dipakai."""
    return _whois is not None


def get_domain_age_days(hostname: str, timeout: float = 4.0):
    """Kembalikan umur domain dalam hari, atau None bila tidak tersedia.

    None dikembalikan pada semua kondisi 'tidak bisa': pustaka tak ada, host
    berupa IP, timeout, rate-limit, domain tidak terdaftar, atau error apa pun.
    Ini disengaja agar pemakaian bersifat situasional dan tidak pernah
    menggagalkan prediksi.
    """
    if not hostname or _whois is None:
        return None
    if _IP_RE.match(hostname):
        return None

    # Ambil domain terdaftar sederhana (dua label terakhir) untuk query WHOIS.
    labels = [l for l in hostname.split(".") if l]
    if len(labels) < 2:
        return None

    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        info = _whois.whois(hostname)
        created = getattr(info, "creation_date", None)
        if isinstance(created, (list, tuple)):
            created = created[0] if created else None
        if not isinstance(created, datetime):
            return None
        age = (datetime.now() - created).days
        return age if age >= 0 else None
    except Exception:
        # Termasuk timeout, rate-limit, parsing gagal, domain tak ditemukan.
        return None
    finally:
        socket.setdefaulttimeout(previous_timeout)


def describe_domain_age(age_days):
    """Susun frasa ramah-awam untuk umur domain (untuk narasi penjelasan).

    Mengembalikan dict {available, text, direction} atau None bila umur tidak
    diketahui. direction: 'phishing' (mencurigakan) / 'aman' / 'netral'.
    """
    if age_days is None:
        return None
    if age_days < 30:
        return {
            "available": True,
            "text": f"Domain baru dibuat sekitar {age_days} hari lalu (sangat muda)",
            "direction": "phishing",
        }
    if age_days < 180:
        months = max(age_days // 30, 1)
        return {
            "available": True,
            "text": f"Domain masih tergolong baru (± {months} bulan)",
            "direction": "phishing",
        }
    years = age_days / 365.0
    if years >= 2:
        return {
            "available": True,
            "text": f"Domain sudah cukup lama terdaftar (± {years:.0f} tahun)",
            "direction": "aman",
        }
    return {
        "available": True,
        "text": f"Domain terdaftar sekitar {age_days} hari",
        "direction": "netral",
    }
