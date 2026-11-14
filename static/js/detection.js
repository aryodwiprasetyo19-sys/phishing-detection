/*
 * detection.js
 * ============
 * Logika interaksi halaman "Periksa URL".
 *
 * Periksa Keamanan -> POST /api/detect { url, mode }. Hasil UTAMA memakai
 * mode "ensemble" (gabungan 4 algoritma), dan menyertakan pendapat
 * masing-masing algoritma secara terpisah. Bila DB aktif -> redirect ke
 * halaman hasil; bila tidak -> render inline (dengan narasi Analisis AI).
 *
 * Bergantung pada shap_chart.js (renderProbabilityGauge, renderShapChart).
 */

(function () {
    "use strict";

    const scanner = document.getElementById("scanner");
    const scanStatus = document.getElementById("scanStatus");
    const urlInput = document.getElementById("urlInput");
    const modeSelect = document.getElementById("modeSelect");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const messageArea = document.getElementById("messageArea");
    const resultSection = document.getElementById("inlineResultSection");
    const inlineResult = document.getElementById("inlineResult");

    if (!analyzeBtn || !urlInput) return;

    function setScanning(active, statusText) {
        if (scanner) scanner.classList.toggle("is-scanning", active);
        if (scanStatus) scanStatus.textContent = statusText || (active ? "memindai..." : "siap");
        analyzeBtn.disabled = active;
        urlInput.disabled = active;
        if (modeSelect) modeSelect.disabled = active;
    }

    function showMessage(text, type) {
        const cls = type === "info" ? "alert-info-soc" : "alert-danger-soc";
        messageArea.innerHTML =
            '<div class="alert-soc ' + cls + '">' + escapeHtml(text) + "</div>";
    }

    function clearMessage() { messageArea.innerHTML = ""; }
    function hideSections() {
        if (resultSection) resultSection.style.display = "none";
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str == null ? "" : String(str);
        return div.innerHTML;
    }

    function readUrl() {
        const url = (urlInput.value || "").trim();
        if (!url) {
            showMessage("Silakan tempel alamat link yang ingin diperiksa.", "danger");
            urlInput.focus();
            return null;
        }
        return url;
    }

    // ---------- Periksa Keamanan ----------
    async function analyze() {
        const url = readUrl();
        clearMessage();
        hideSections();
        if (!url) return;

        const mode = modeSelect ? modeSelect.value : "ensemble";
        setScanning(true, "memeriksa keamanan...");

        try {
            const resp = await fetch("/api/detect", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url: url, mode: mode }),
            });
            const data = await resp.json();

            if (!resp.ok || !data.success) {
                setScanning(false, "gagal");
                showMessage((data && data.error) || "Terjadi kesalahan saat memproses.", "danger");
                return;
            }
            if (data.redirect_url) {
                setScanning(true, "membuka hasil...");
                window.location.href = data.redirect_url;
                return;
            }
            if (data.result) {
                setScanning(false, "selesai");
                renderInline(data.result);
                return;
            }
            setScanning(false, "gagal");
            showMessage("Respons server tidak dikenali.", "danger");
        } catch (err) {
            setScanning(false, "gagal");
            showMessage("Gagal terhubung ke server: " + err.message, "danger");
        }
    }

    function verdictMod(risk) {
        return risk === "High" ? "high" : risk === "Medium" ? "med" : "low";
    }
    function statusText(label) {
        return label === "phishing" ? "Berpotensi Phishing" : "Kemungkinan Aman";
    }

    // Satu baris "pendapat algoritma" dalam breakdown 4 algoritma.
    function compareRow(m, isBest) {
        return (
            '<div class="compare-verdict-item">' +
                '<div class="compare-verdict">' +
                    '<span class="compare-verdict__name">' + escapeHtml(m.model_display) +
                        (isBest ? ' <span class="risk-pill Low">terbaik</span>' : "") + "</span>" +
                    '<span class="label-pill ' + escapeHtml(m.prediction_label) + '">' + statusText(m.prediction_label) + "</span>" +
                    '<span class="risk-pill ' + escapeHtml(m.risk_level) + '">' + escapeHtml(m.risk_level) + "</span>" +
                    '<span class="mono">' + m.phishing_probability_pct + "%</span>" +
                "</div>" +
                '<p class="compare-verdict__explain">' + escapeHtml(m.explanation || "") + "</p>" +
            "</div>"
        );
    }

    // Render hasil utama secara inline (mode fallback tanpa database).
    function renderInline(r) {
        const vmod = verdictMod(r.risk_level);
        const label = statusText(r.prediction_label);
        const probPct = r.phishing_probability_pct;

        // Pendapat masing-masing algoritma (bukan cuma hasil ensemble/mode terpilih).
        let perModelHtml = "";
        if (r.per_model && r.per_model.length) {
            const bestName = r.best_model ? r.best_model.model_name : null;
            let rows = "";
            r.per_model.forEach(function (m) {
                rows += compareRow(m, bestName && m.model_used === bestName);
            });
            perModelHtml =
                '<div class="card-soc mb-4"><div class="card-body">' +
                    '<div class="eyebrow mb-1">PERBANDINGAN 4 ALGORITMA</div>' +
                    '<h3 class="mb-1">Pendapat masing-masing algoritma</h3>' +
                    '<p class="text-muted mb-3" style="font-size:.9rem;">Hasil di atas adalah <b>' +
                    escapeHtml(r.model_display) + '</b>. Berikut keputusan tiap algoritma secara terpisah untuk URL yang sama.</p>' +
                    rows +
                "</div></div>";
        }

        // Narasi Analisis AI (berbahasa awam).
        let narrativeHtml = "";
        if (r.narrative) {
            let items = "";
            (r.narrative.points || []).forEach(function (p) {
                items += '<li class="narrative-item narrative-item--' + p.direction + '">' +
                    escapeHtml(p.text) + "</li>";
            });
            let bestModelHtml = "";
            if (r.best_model) {
                bestModelHtml =
                    '<p class="text-muted mt-3 mb-0" style="font-size:.85rem;">📊 Analisis ini dihasilkan dari <b>' +
                    escapeHtml(r.model_display) + '</b>. Sistem melatih 4 algoritma secara paralel; ' +
                    'berdasarkan evaluasi pada data uji, algoritma dengan performa terbaik adalah <b>' +
                    escapeHtml(r.best_model.display_name) + '</b> (F1-Score ' +
                    Number(r.best_model.f1_score).toFixed(4) + ', Akurasi ' +
                    (Number(r.best_model.accuracy) * 100).toFixed(2) + '%).</p>';
            }
            narrativeHtml =
                '<div class="card-soc mb-4 narrative-card narrative-card--' + vmod + '"><div class="card-body">' +
                    '<div class="eyebrow mb-2">🧠 ANALISIS AI</div>' +
                    '<ul class="narrative-list">' + items + "</ul>" +
                    '<p class="narrative-conclusion">' + escapeHtml(r.narrative.conclusion) + "</p>" +
                    bestModelHtml +
                "</div></div>";
        }

        inlineResult.innerHTML =
            '<div class="verdict verdict--' + vmod + '">' +
                '<span class="verdict__badge badge-' + vmod + '">' + escapeHtml(label) + "</span>" +
                '<div class="flex-grow-1"><div style="font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;">Tingkat Risiko</div>' +
                    '<div style="font-family:var(--font-display);font-size:1.35rem;font-weight:700;">' +
                    escapeHtml(r.risk_level) + " &middot; " + probPct + "% kemungkinan phishing</div></div>" +
            "</div>" +
            '<div class="card-soc my-4"><div class="card-body">' +
                '<div style="font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;" class="mb-2">Alamat yang diperiksa</div>' +
                '<div class="url-display mb-2">' + escapeHtml(r.url) + "</div>" +
                '<div class="d-flex flex-wrap gap-3" style="font-size:.9rem;">' +
                    "<span>Analisis: <b>" + escapeHtml(r.model_display) + "</b></span>" +
                    '<span>Status: <span class="label-pill ' + escapeHtml(r.prediction_label) + '">' + escapeHtml(label) + "</span></span>" +
                    '<span>Risiko: <span class="risk-pill ' + escapeHtml(r.risk_level) + '">' + escapeHtml(r.risk_level) + "</span></span>" +
                "</div></div></div>" +
            perModelHtml +
            narrativeHtml +
            '<div class="alert-soc alert-info-soc mb-4"><strong>Saran:</strong> ' + escapeHtml(r.suggestion) + "</div>" +
            '<div class="card-soc"><div class="card-body">' +
                '<details class="tech-details"><summary>Lihat grafik faktor teknis (SHAP)</summary>' +
                '<div style="position:relative;height:320px;" class="mt-3"><canvas id="inlineShap"></canvas></div>' +
                "</details></div></div>";

        resultSection.style.display = "block";
        renderShapChart("inlineShap", r.top_features || []);
        resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    analyzeBtn.addEventListener("click", analyze);
    urlInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") analyze();
    });
})();
