"""
routes/detection_routes.py
==========================
Blueprint untuk fitur inti: deteksi URL dan tampilan hasil analisis.

Endpoint:
  GET  /detect            -> halaman form input URL (ramah awam).
  POST /api/detect        -> deteksi JSON. mode: "ensemble" (gabungan 4 algoritma,
                             default) atau salah satu kunci algoritma. Hasil selalu
                             menyertakan pendapat masing-masing algoritma (per_model).
  GET  /result/<id>       -> halaman hasil analisis lengkap dari DB.

Prinsip: hasil UTAMA adalah gabungan (ensemble) keempat algoritma, ditampilkan
berdampingan dengan keputusan tiap algoritma secara terpisah. Kegagalan
database tidak pernah membatalkan hasil prediksi.
"""

from flask import Blueprint, jsonify, render_template, request, url_for

from config import MODEL_NAMES
from ml import model_loader
from ml.predictor import (
    ENSEMBLE_DISPLAY,
    ENSEMBLE_KEY,
    FEATURE_LABELS_ID,
    build_explanation_reasons,
    build_narrative,
    predict,
    score_all_models,
)
from utils import whois_lookup
from utils.validators import validate_url

detection_bp = Blueprint("detection", __name__)


def _mode_choices():
    """Daftar pilihan mode untuk dropdown: ensemble dulu, lalu tiap algoritma."""
    choices = [(ENSEMBLE_KEY, ENSEMBLE_DISPLAY)]
    for key in model_loader.available_models():
        choices.append((key, MODEL_NAMES.get(key, key)))
    return choices


@detection_bp.route("/detect")
def detect():
    """Halaman form input URL."""
    return render_template(
        "detection.html",
        mode_choices=_mode_choices(),
        ensemble_key=ENSEMBLE_KEY,
        available_count=len(model_loader.available_models()),
    )


@detection_bp.route("/api/detect", methods=["POST"])
def api_detect():
    """Deteksi satu URL (default: gabungan 4 algoritma)."""
    data = request.get_json(silent=True) or request.form
    url = (data.get("url") or "").strip()
    mode = (data.get("mode") or ENSEMBLE_KEY).strip()

    is_valid, result_or_msg = validate_url(url)
    if not is_valid:
        return jsonify({"success": False, "error": result_or_msg}), 400
    url = result_or_msg

    if not model_loader.available_models():
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Model belum tersedia. Jalankan notebook/pipeline "
                    "training terlebih dahulu (lihat README).",
                }
            ),
            503,
        )

    try:
        result = predict(url, mode)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "error": f"Gagal memproses: {exc}"}), 500

    # Simpan ke DB (opsional).
    prediction_id = None
    try:
        from database.models import insert_prediction

        prediction_id = insert_prediction(result)
    except Exception:
        prediction_id = None

    if prediction_id is not None:
        return jsonify(
            {
                "success": True,
                "prediction_id": prediction_id,
                "redirect_url": url_for(
                    "detection.result", prediction_id=prediction_id
                ),
            }
        )

    return jsonify({"success": True, "prediction_id": None, "result": result})


@detection_bp.route("/result/<int:prediction_id>")
def result(prediction_id: int):
    """Halaman hasil analisis lengkap untuk satu prediksi tersimpan."""
    try:
        from database.models import get_prediction_by_id

        record = get_prediction_by_id(prediction_id)
    except Exception:
        record = None

    if not record:
        return render_template("404.html"), 404

    shap_values = record.get("shap_values") or {}
    top_features = (
        sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        if shap_values
        else []
    )

    record["phishing_probability_pct"] = round(
        record["phishing_probability"] * 100, 2
    )
    used = record.get("model_used")
    record["model_display"] = (
        ENSEMBLE_DISPLAY if used == ENSEMBLE_KEY else MODEL_NAMES.get(used, used)
    )
    # Info algoritma terbaik tidak disimpan per-prediksi di DB (bersifat
    # evaluasi global, bukan hasil prediksi ini) -- selalu ambil terbaru.
    record["best_model"] = model_loader.best_model()

    features = record.get("extracted_features") or {}
    reasons = build_explanation_reasons(
        top_features, features, signed=bool(shap_values)
    )

    # Narasi awam: bangun ulang dari fitur tersimpan + umur domain tersimpan.
    age_days = record.get("domain_age_days")
    domain_age_info = whois_lookup.describe_domain_age(age_days)
    narrative = build_narrative(
        features, record["prediction_label"], record["risk_level"], domain_age_info,
        shap_dict=shap_values or None,
    )

    # Pendapat masing-masing algoritma -- hanya relevan bila prediksi ini
    # memakai mode gabungan (ensemble); bila hanya satu algoritma yang
    # dipakai, tidak ada breakdown yang perlu ditampilkan.
    per_model = []
    if used == ENSEMBLE_KEY:
        try:
            per_model = score_all_models(record["url"], features, age_days)
        except Exception:
            per_model = []

    return render_template(
        "result.html",
        r=record,
        top_features=top_features,
        reasons=reasons,
        narrative=narrative,
        feature_labels=FEATURE_LABELS_ID,
        per_model=per_model,
    )
