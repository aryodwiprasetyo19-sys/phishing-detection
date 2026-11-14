"""
database/db.py
==============
Manajemen koneksi ke MySQL menggunakan PyMySQL.

Memakai koneksi per-permintaan yang dibuka dan ditutup lewat context manager
(`with get_connection() as conn:`). Pendekatan ini sederhana, aman terhadap
kebocoran koneksi, dan sepenuhnya memadai untuk beban aplikasi skripsi.

Semua kredensial diambil dari config.DB_CONFIG (yang bersumber dari .env).
"""

from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor

from config import DB_CONFIG


def _connect(include_db: bool = True):
    """Buat koneksi PyMySQL baru.

    Parameters
    ----------
    include_db : bool
        Jika False, koneksi dibuat tanpa memilih database tertentu
        (dibutuhkan saat membuat database untuk pertama kali).
    """
    params = dict(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        charset=DB_CONFIG["charset"],
        cursorclass=DictCursor,
        autocommit=False,
    )
    if include_db:
        params["database"] = DB_CONFIG["database"]
    return pymysql.connect(**params)


@contextmanager
def get_connection(include_db: bool = True):
    """Context manager koneksi DB.

    Otomatis melakukan commit bila blok selesai tanpa error, rollback bila
    terjadi exception, dan selalu menutup koneksi.
    """
    conn = _connect(include_db=include_db)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def test_connection() -> bool:
    """Cek apakah koneksi ke database berhasil. Return True/False."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[db] Koneksi database gagal: {exc}")
        return False
