/**
 * AI Cyclone Early Warning System - Dashboard JavaScript
 * Handles: Map, Charts, Image Upload, Track Prediction, Risk, Metrics
 * All element IDs match dashboard.html exactly.
 */

// ==========================================
// STATE
// ==========================================
let map, historicalLayer, predictedLayer, uncertaintyLayer, currentMarker;
let charts = { wind: null, pressure: null, lat: null, lon: null };
let isDemoMode = false;
let selectedFile = null;

// ==========================================
// INIT
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    setInterval(updateSysTime, 1000);
    updateSysTime();
    initMap();
    initCharts();
    setupUploadHandler();
    setupTrackPrediction();
    checkSystemMode();
    loadCyclones();
    loadModelMetrics();
});

function updateSysTime() {
    const el = document.getElementById('sysTime');
    if (el) el.textContent = new Date().toLocaleString();
}

// ==========================================
// TOAST
// ==========================================
function showToast(message, type = 'info') {
    const container = document.querySelector('.toast-container');
    if (!container) return;
    const id = 'toast-' + Date.now();
    const borderColor = type === 'error' ? 'var(--accent-red, #ff3366)' : 'var(--accent-cyan, #00d4ff)';
    const title = type === 'error' ? 'Error' : 'Notification';
    const html = `
        <div id="${id}" class="toast" role="alert" style="background:rgba(16,24,43,0.95);color:#fff;border-left:4px solid ${borderColor};">
            <div class="toast-header" style="background:rgba(0,0,0,0.3);color:#fff;border-bottom:1px solid rgba(255,255,255,0.1);">
                <strong class="me-auto">${title}</strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
            </div>
            <div class="toast-body">${message}</div>
        </div>`;
    container.insertAdjacentHTML('beforeend', html);
    const toastEl = document.getElementById(id);
    new bootstrap.Toast(toastEl, { delay: 5000 }).show();
    toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
}

// ==========================================
// SYSTEM MODE
// ==========================================
function checkSystemMode() {
    fetch('/api/status')
        .then(r => r.json())
        .catch(() => ({ demo_mode: true }))
        .then(data => {
            isDemoMode = data.demo_mode !== false;
            const badge = document.getElementById('demoBadge');
            if (badge) badge.style.display = isDemoMode ? 'inline-block' : 'none';
        });
}

// ==========================================
// MAP MODULE (Leaflet.js)
// ==========================================
function initMap() {
    map = L.map('cyclone-map').setView([15.0, 85.0], 5);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
        subdomains: 'abcd', maxZoom: 19
    }).addTo(map);
}

function updateMap(historicalData, predictedData) {
    if (historicalLayer) map.removeLayer(historicalLayer);
    if (predictedLayer) map.removeLayer(predictedLayer);
    if (uncertaintyLayer) map.removeLayer(uncertaintyLayer);
    if (currentMarker) map.removeLayer(currentMarker);
    historicalLayer = predictedLayer = uncertaintyLayer = currentMarker = null;

    const allPts = [];

    // Historical track
    if (historicalData && historicalData.length > 0) {
        const latlngs = historicalData.map(p => { allPts.push([p.lat, p.lon]); return [p.lat, p.lon]; });
        historicalLayer = L.featureGroup().addTo(map);
        L.polyline(latlngs, { color: '#ff3366', weight: 3, opacity: 0.8 }).addTo(historicalLayer);
        historicalData.forEach((p, i) => {
            const isLast = i === historicalData.length - 1;
            if (isLast) {
                const icon = L.divIcon({
                    className: '', html: "<div style='background:#ff3366;width:14px;height:14px;border-radius:50%;box-shadow:0 0 12px #ff3366;'></div>",
                    iconSize: [14, 14], iconAnchor: [7, 7]
                });
                currentMarker = L.marker([p.lat, p.lon], { icon }).addTo(map);
                currentMarker.bindPopup(`<b>Current Position</b><br>Lat: ${p.lat}, Lon: ${p.lon}<br>Wind: ${p.wind ?? 'N/A'} kt<br>Pressure: ${p.pressure ?? 'N/A'} hPa`);
            } else {
                L.circleMarker([p.lat, p.lon], { radius: 4, color: '#ff3366', fillColor: '#ff3366', fillOpacity: 0.8 })
                    .addTo(historicalLayer)
                    .bindPopup(`Time: ${p.time ?? 'N/A'}<br>Lat: ${p.lat}, Lon: ${p.lon}<br>Wind: ${p.wind ?? 'N/A'}<br>Pressure: ${p.pressure ?? 'N/A'}`);
            }
        });
    }

    // Predicted track
    if (predictedData && predictedData.length > 0) {
        const predLatlngs = predictedData.map(p => { allPts.push([p.lat, p.lon]); return [p.lat, p.lon]; });
        if (historicalData && historicalData.length > 0) {
            const last = historicalData[historicalData.length - 1];
            predLatlngs.unshift([last.lat, last.lon]);
        }
        predictedLayer = L.featureGroup().addTo(map);
        L.polyline(predLatlngs, { color: '#ff6b35', weight: 3, dashArray: '8,8' }).addTo(predictedLayer);
        predictedData.forEach(p => {
            const norm = normalizePoint(p);
            L.circleMarker([p.lat, p.lon], { radius: 4, color: '#ff6b35', fillColor: '#10182b', fillOpacity: 1, weight: 2 })
                .addTo(predictedLayer)
                .bindPopup(`<b>Predicted T+${norm.time_offset}h</b><br>Lat: ${p.lat.toFixed(2)}, Lon: ${p.lon.toFixed(2)}<br>Wind: ${norm.wind_speed ?? 'N/A'}<br>Pressure: ${norm.pressure ?? 'N/A'}`);
        });

        // Uncertainty cone
        const cone = predictedData.filter(p => Number.isFinite(Number(p.uncertainty_radius_km)));
        if (cone.length > 1) {
            const left = [], right = [];
            cone.forEach(p => {
                const r = Number(p.uncertainty_radius_km) / 111;
                left.push([p.lat + r, p.lon]);
                right.unshift([p.lat - r, p.lon]);
            });
            uncertaintyLayer = L.polygon(left.concat(right), { color: '#f4c95d', fillColor: '#f4c95d', fillOpacity: 0.12, weight: 1 }).addTo(map);
        }
    }

    if (allPts.length > 0) map.fitBounds(L.latLngBounds(allPts), { padding: [50, 50] });
}

// ==========================================
// IMAGE UPLOAD
// ==========================================
function setupUploadHandler() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const btnAnalyze = document.getElementById('btnAnalyzeImg');
    if (!dropZone || !fileInput || !btnAnalyze) return;

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault(); dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', function () {
        if (this.files.length) handleFileSelect(this.files[0]);
    });

    function handleFileSelect(file) {
        if (!file.type.match('image.*')) { showToast('Please select a valid image file.', 'error'); return; }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            document.getElementById('imagePreview').src = e.target.result;
            document.getElementById('imagePreviewContainer').classList.remove('d-none');
            dropZone.classList.add('d-none');
            btnAnalyze.classList.remove('disabled');
        };
        reader.readAsDataURL(file);
    }

    btnAnalyze.addEventListener('click', () => {
        if (!selectedFile || btnAnalyze.classList.contains('disabled')) return;
        const origText = btnAnalyze.innerHTML;
        btnAnalyze.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';
        btnAnalyze.classList.add('disabled');

        const formData = new FormData();
        formData.append('file', selectedFile);

        fetch('/predict/image', { method: 'POST', body: formData })
            .then(r => r.json())
            .catch(() => ({ success: false, error: 'Image analysis service unavailable.' }))
            .then(data => {
                const resultsPanel = document.getElementById('imageAnalysisResults');
                if (data.success !== false) {
                    const confidence = data.confidence_percent ?? ((data.confidence || 0) * 100);
                    setText('card-confidence', confidence.toFixed(1) + '%');
                    setWidth('confidence-bar', confidence + '%');
                    setText('res-cyclone-detected', data.cyclone_detected === true ? 'YES' : (data.cyclone_detected === false ? 'NO' : 'N/A'));
                    setText('res-prediction', data.prediction || data.class_name || '--');
                    setText('res-confidence', confidence.toFixed(1) + '%');
                    setText('res-risk-score', data.image_risk_score != null ? (data.image_risk_score + ' / 100') : 'N/A');
                    setText('res-risk-level', data.image_risk_level || 'N/A');
                    if (resultsPanel) resultsPanel.classList.remove('d-none');

                    const identification = data.cyclone_identification || {};
                    if (identification.matched) {
                        predictTrack(identification.cyclone_id, true);
                        showToast('Cyclone identified: ' + (identification.cyclone_name || identification.cyclone_id) + '. Loading track...');
                    } else {
                        const selectedCyclone = document.getElementById('cycloneSelect')?.value;
                        if (selectedCyclone) {
                            setText('res-track-status', 'Manual cyclone selection: loading LSTM track forecast. The image itself was not used to identify this storm.');
                            predictTrack(selectedCyclone, true, true);
                        } else {
                            setText('res-track-status', 'This image has no verified storm mapping, so no track was assigned. Choose a named historical cyclone below only for manual exploration.');
                            showToast('CNN analysis is complete. Select the cyclone to run the LSTM forecast.', 'info');
                        }
                    }
                } else {
                    if (resultsPanel) resultsPanel.classList.add('d-none');
                    showToast(data.error || 'Analysis failed.', 'error');
                }
            })
            .finally(() => { btnAnalyze.innerHTML = origText; btnAnalyze.classList.remove('disabled'); });
    });
}

// ==========================================
// TRACK PREDICTION
// ==========================================
function loadCyclones() {
    fetch('/api/cyclones')
        .then(r => r.json())
        .catch(() => [])
        .then(data => {
            const select = document.getElementById('cycloneSelect');
            if (!select) return;
            const cyclones = Array.isArray(data) ? data : [];
            // Keep default option
            select.innerHTML = '<option value="">-- Select Cyclone --</option>';
            cyclones.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c.id;
                opt.textContent = c.name || c.id;
                select.appendChild(opt);
            });
        });
}

function setupTrackPrediction() {
    const btn = document.getElementById('btnPredictTrack');
    if (!btn) return;
    btn.addEventListener('click', () => {
        const cid = document.getElementById('cycloneSelect')?.value;
        if (cid) predictTrack(cid);
        else showToast('Please select a cyclone first.', 'error');
    });
}

function predictTrack(cycloneId, automatic = false, manuallyAssociated = false) {
    const btn = document.getElementById('btnPredictTrack');
    const origText = btn ? btn.innerHTML : '';
    if (btn) { btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Predicting...'; btn.disabled = true; }

    fetch('/predict/track', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cyclone_id: cycloneId })
    })
        .then(r => r.ok ? r.json() : r.json().then(d => Promise.reject(new Error(d.error || 'Track prediction failed.'))))
        .then(data => {
            const historical = (data.historical || []).map(normalizePoint);
            const predicted = (data.predicted || []).map(normalizePoint);
            updateMap(historical, predicted);
            updateTable(predicted);
            updateChartsData(historical, predicted);

            setText('card-cyclone-name', data.cyclone_name || data.cyclone_id || 'Unknown');
            const current = historical.length ? historical[historical.length - 1] : {};
            setText('card-wind', (current.wind_speed ?? '--') + ' km/h');
            setText('card-pressure', (current.pressure ?? '--') + ' hPa');

            const finalPoint = predicted[predicted.length - 1];
            const uncertainty = Number(finalPoint?.uncertainty_radius_km);
            const uncertaintyText = Number.isFinite(uncertainty)
                ? `LSTM forecast loaded. T+48h uncertainty corridor: ±${Math.round(uncertainty)} km (75th-percentile held-out error).`
                : (data.demo_mode ? 'Demo trajectory loaded; it is not an operational forecast.' : 'LSTM forecast loaded; uncertainty data is unavailable.');
            setText('res-track-status', manuallyAssociated
                ? `Manual storm association. ${uncertaintyText}`
                : uncertaintyText);

            return fetch('/api/risk').then(r => r.json());
        })
        .then(assessRisk)
        .then(() => showToast(automatic ? 'Track loaded from identified cyclone.' : 'Track prediction updated.'))
        .catch(err => showToast(err.message, 'error'))
        .finally(() => { if (btn) { btn.innerHTML = origText; btn.disabled = false; } });
}

function normalizePoint(p) {
    return {
        ...p,
        wind_speed: p.wind_speed ?? p.wind ?? p.wind_estimated ?? null,
        pressure: p.pressure ?? p.pressure_estimated ?? null,
        time_offset: p.time_offset ?? p.time ?? 'N/A'
    };
}

// ==========================================
// TABLE
// ==========================================
function updateTable(predictions) {
    const tbody = document.getElementById('predicted-track-body');
    if (!tbody) return;
    if (!predictions || predictions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-4">No prediction data available</td></tr>';
        return;
    }
    tbody.innerHTML = '';
    predictions.forEach(p => {
        const ws = p.wind_speed;
        let badge = 'badge bg-success';
        if (ws !== null && ws !== undefined) {
            if (ws >= 90) badge = 'badge bg-danger';
            else if (ws >= 63) badge = 'badge bg-warning text-white';
            else if (ws >= 34) badge = 'badge bg-info';
        }
        const riskLabel = ws == null ? '--' : (ws >= 90 ? 'VERY HIGH' : ws >= 63 ? 'HIGH' : ws >= 34 ? 'MODERATE' : 'LOW');
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>+${p.time_offset}h</td>
            <td>${Number(p.lat).toFixed(2)}°</td>
            <td>${Number(p.lon).toFixed(2)}°</td>
            <td>${ws != null ? Number(ws).toFixed(1) : '--'}</td>
            <td>${p.pressure != null ? Number(p.pressure).toFixed(1) : '--'}</td>
            <td><span class="${badge}">${riskLabel}</span></td>`;
        tbody.appendChild(tr);
    });
}

// ==========================================
// RISK ASSESSMENT
// ==========================================
function assessRisk(risk) {
    if (!risk || !risk.risk_level) return;
    const level = risk.risk_level;
    const colorMap = { 'LOW': 'success', 'MODERATE': 'warning', 'HIGH': 'warning', 'VERY HIGH': 'danger' };
    const bsColor = colorMap[level] || 'secondary';

    // KPI card
    const cardRisk = document.getElementById('card-risk');
    if (cardRisk) {
        const iconEl = document.getElementById('risk-icon');
        const iconClass = level === 'LOW' ? 'fa-shield-halved' : 'fa-triangle-exclamation';
        if (iconEl) iconEl.className = `fas ${iconClass} me-2`;
        cardRisk.className = `fs-5 fw-bold text-${bsColor}`;
        // Preserve the icon
        if (iconEl) cardRisk.prepend(iconEl);
        cardRisk.appendChild(document.createTextNode(level));
    }

    // Warning panel
    const panelIcon = document.getElementById('panel-risk-icon');
    if (panelIcon) panelIcon.className = `fas ${level === 'LOW' ? 'fa-shield-halved' : 'fa-triangle-exclamation'} me-2 text-${bsColor}`;
    setText('panel-risk-level', level + ' RISK');
    setText('panel-risk-score', risk.risk_score ?? '--');
    setText('panel-risk-source', risk.risk_source || 'Meteorological analysis');
    setText('panel-warning-desc', risk.reason || 'Assessment pending.');

    // Recommended actions
    const actionsList = document.getElementById('panel-actions-list');
    if (actionsList && risk.recommended_actions) {
        actionsList.innerHTML = '';
        risk.recommended_actions.forEach(act => {
            const li = document.createElement('li');
            li.textContent = act;
            actionsList.appendChild(li);
        });
    }
}

// ==========================================
// CHARTS (Chart.js)
// ==========================================
function initCharts() {
    Chart.defaults.color = '#a0aab2';
    Chart.defaults.font.family = "'Inter', sans-serif";
    const gridColor = 'rgba(255,255,255,0.06)';
    const opts = (title) => ({
        responsive: true, maintainAspectRatio: false,
        plugins: { title: { display: true, text: title, color: '#fff', font: { size: 13 } }, legend: { display: true, labels: { boxWidth: 12 } } },
        scales: { x: { grid: { color: gridColor } }, y: { grid: { color: gridColor } } }
    });

    charts.wind = new Chart(document.getElementById('windChart'), { type: 'line', data: { labels: [], datasets: [] }, options: opts('Wind Speed (km/h)') });
    charts.pressure = new Chart(document.getElementById('pressureChart'), { type: 'line', data: { labels: [], datasets: [] }, options: opts('Pressure (hPa)') });
    charts.lat = new Chart(document.getElementById('latChart'), { type: 'line', data: { labels: [], datasets: [] }, options: opts('Latitude') });
    charts.lon = new Chart(document.getElementById('lonChart'), { type: 'line', data: { labels: [], datasets: [] }, options: opts('Longitude') });
}

function updateChartsData(hist, pred) {
    const labels = [];
    const windH = [], presH = [], latH = [], lonH = [];

    hist.forEach((h, i) => {
        labels.push(`H-${hist.length - 1 - i}`);
        windH.push(h.wind_speed); presH.push(h.pressure); latH.push(h.lat); lonH.push(h.lon);
    });

    // Bridge: predicted line starts from last historical point
    const bridgeLen = hist.length > 0 ? hist.length - 1 : 0;
    const windP = new Array(bridgeLen).fill(null);
    const presP = new Array(bridgeLen).fill(null);
    const latP = new Array(bridgeLen).fill(null);
    const lonP = new Array(bridgeLen).fill(null);
    if (hist.length > 0) {
        const last = hist[hist.length - 1];
        windP.push(last.wind_speed); presP.push(last.pressure); latP.push(last.lat); lonP.push(last.lon);
    }
    pred.forEach(p => {
        labels.push(`+${p.time_offset}h`);
        windP.push(p.wind_speed); presP.push(p.pressure); latP.push(p.lat); lonP.push(p.lon);
    });

    const histDS = (data) => ({ label: 'Historical', data, borderColor: '#ff3366', backgroundColor: 'rgba(255,51,102,0.1)', tension: 0.3, fill: true, pointRadius: 2 });
    const predDS = (data) => ({ label: 'Predicted', data, borderColor: '#ff6b35', borderDash: [5, 5], tension: 0.3, pointRadius: 2 });

    charts.wind.data = { labels, datasets: [histDS(windH), predDS(windP)] }; charts.wind.update();
    charts.pressure.data = { labels, datasets: [histDS(presH), predDS(presP)] }; charts.pressure.update();
    charts.lat.data = { labels, datasets: [histDS(latH), predDS(latP)] }; charts.lat.update();
    charts.lon.data = { labels, datasets: [histDS(lonH), predDS(lonP)] }; charts.lon.update();
}

// ==========================================
// MODEL METRICS
// ==========================================
function loadModelMetrics() {
    // The HTML has hardcoded metrics placeholders. We'll update them if dynamic IDs exist.
    // Fall back silently if elements don't exist.
    fetch('/api/model-metrics')
        .then(r => r.json())
        .catch(() => ({ available: false }))
        .then(data => {
            if (data.available === false) return;
            // These IDs may not exist in current HTML — that's OK, setText handles null gracefully
            const cnn = data.cnn || {};
            const lstm = data.lstm || {};
            setText('metric-lat', lstm.latitude_mae != null ? lstm.latitude_mae.toFixed(2) + '°' : (lstm.lat_mae != null ? lstm.lat_mae + '°' : null));
            setText('metric-lon', lstm.longitude_mae != null ? lstm.longitude_mae.toFixed(2) + '°' : (lstm.lon_mae != null ? lstm.lon_mae + '°' : null));
            setText('metric-acc', cnn.accuracy != null ? (cnn.accuracy * 100).toFixed(1) + '%' : null);
            setText('metric-f1', cnn.f1_score != null ? (cnn.f1_score * 100).toFixed(1) + '%' : null);
        });
}

// ==========================================
// HELPERS
// ==========================================
function setText(id, text) {
    const el = document.getElementById(id);
    if (el && text != null) el.textContent = text;
}
function setWidth(id, width) {
    const el = document.getElementById(id);
    if (el) el.style.width = width;
}
