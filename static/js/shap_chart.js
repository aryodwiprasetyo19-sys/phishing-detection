/* =========================================================================
   shap_chart.js
   Fungsi render grafik berbasis Chart.js:
     - renderShapChart(): bar chart horizontal kontribusi fitur (SHAP).
     - renderProbabilityGauge(): doughnut sebagai gauge probabilitas phishing.
   Dipakai baik pada hasil inline (detection.js) maupun halaman result.html.
   ========================================================================= */

// Label fitur yang lebih ramah dibaca (fallback ke nama asli bila tak ada).
const FEATURE_LABELS = {
    url_length: "Panjang URL",
    hostname_length: "Panjang hostname",
    path_length: "Panjang path",
    query_length: "Panjang query",
    count_dots: "Jumlah titik",
    count_hyphen: "Jumlah tanda hubung",
    count_underscore: "Jumlah underscore",
    count_slash: "Jumlah slash",
    count_question: "Jumlah '?'",
    count_equal: "Jumlah '='",
    count_at: "Jumlah '@'",
    count_and: "Jumlah '&'",
    count_percent: "Jumlah '%'",
    count_digits: "Jumlah digit",
    count_letters: "Jumlah huruf",
    count_params: "Jumlah parameter",
    path_depth: "Kedalaman path",
    num_subdomains: "Jumlah subdomain",
    has_ip: "Memakai IP",
    has_https: "Memakai HTTPS",
    has_port: "Memakai port",
    has_shortener: "URL shortener",
    has_suspicious_tld: "TLD mencurigakan",
    has_double_slash_in_path: "Double slash di path",
    has_at_symbol: "Simbol '@'",
    http_in_path: "'http' di path",
    digit_letter_ratio: "Rasio digit/huruf",
    special_char_ratio: "Rasio karakter khusus",
    longest_word_length: "Kata terpanjang",
    url_entropy: "Entropi URL",
    contains_brand_name: "Mengandung nama brand",
    contains_login_keyword: "Kata kunci 'login'",
    contains_verify_keyword: "Kata kunci 'verify'",
    contains_secure_keyword: "Kata kunci 'secure'",
    contains_account_keyword: "Kata kunci 'account'",
    contains_payment_keyword: "Kata kunci pembayaran",
    contains_update_keyword: "Kata kunci 'update'",
    contains_wallet_keyword: "Kata kunci dompet/kripto",
    brand_in_subdomain: "Brand di subdomain",
    brand_not_in_registered_domain: "Brand di luar domain terdaftar",
};

function prettyFeature(name) {
    return FEATURE_LABELS[name] || name;
}

/**
 * Render bar chart horizontal kontribusi fitur.
 * @param {string} canvasId - id elemen <canvas>.
 * @param {Array} topFeatures - array [ [nama, nilai], ... ] terurut menurun.
 */
function renderShapChart(canvasId, topFeatures) {
    const el = document.getElementById(canvasId);
    if (!el || !topFeatures || topFeatures.length === 0) return;

    const labels = topFeatures.map(f => prettyFeature(f[0]));
    const values = topFeatures.map(f => f[1]);
    // Nilai positif mendorong ke arah phishing (merah), negatif ke arah aman (hijau).
    const colors = values.map(v => (v >= 0 ? "#DC2626" : "#16A34A"));

    new Chart(el, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Kontribusi terhadap prediksi",
                data: values,
                backgroundColor: colors,
                borderRadius: 4,
            }],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `Kontribusi: ${Number(ctx.parsed.x).toFixed(4)}`,
                    },
                },
            },
            scales: {
                x: { grid: { color: "#EEF1F7" }, title: { display: true, text: "← lebih aman   |   lebih phishing →" } },
                y: { grid: { display: false }, ticks: { font: { size: 11 } } },
            },
        },
    });
}

/**
 * Render gauge probabilitas phishing (doughnut setengah lingkaran).
 * @param {string} canvasId
 * @param {number} probPct - probabilitas phishing dalam persen (0-100).
 */
function renderProbabilityGauge(canvasId, probPct) {
    const el = document.getElementById(canvasId);
    if (!el) return;

    let color = "#16A34A";
    if (probPct >= 70) color = "#DC2626";
    else if (probPct >= 30) color = "#D97706";

    new Chart(el, {
        type: "doughnut",
        data: {
            datasets: [{
                data: [probPct, 100 - probPct],
                backgroundColor: [color, "#EEF1F7"],
                borderWidth: 0,
                circumference: 180,
                rotation: 270,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "72%",
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
        },
    });
}
