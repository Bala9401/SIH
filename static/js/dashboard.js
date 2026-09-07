let selectedFile = null;
let map = null;
let layers = [];
let charts = {};

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('sidebar-toggle').addEventListener('click', () => document.getElementById('sidebar').classList.toggle('open'));
    setupUpload();
    setupManualMode();
    initMap();
    loadStatus();
    loadMetrics();
    loadCyclones();
    renderHistory();
});

function setupUpload() {
    const zone = document.getElementById('dropZone');
    const input = document.getElementById('fileInput');
    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', event => { event.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', event => { event.preventDefault(); zone.classList.remove('drag-over'); if (event.dataTransfer.files[0]) selectFile(event.dataTransfer.files[0]); });
    input.addEventListener('change', () => { if (input.files[0]) selectFile(input.files[0]); });
    document.getElementById('remove-file').addEventListener('click', clearFile);
    document.getElementById('analyze-button').addEventListener('click', analyze);
}

function selectFile(file) {
    if (!/\.(jpe?g|png|tiff?|webp)$/i.test(file.name)) return notify('Choose a JPG, PNG, TIFF, or WEBP image.');
    if (file.size > 16 * 1024 * 1024) return notify('The image must be smaller than 16 MB.');
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = event => {
        document.getElementById('imagePreview').src = event.target.result;
        document.getElementById('file-name').textContent = file.name;
        document.getElementById('file-meta').textContent = `${formatBytes(file.size)} | ${file.type || 'image'}`;
        document.getElementById('preview-wrap').classList.remove('hidden');
        document.getElementById('dropZone').classList.add('hidden');
        document.getElementById('analyze-button').disabled = false;
        setStage('ingest', 'READY');
        setText('quality-image', 'VALID');
    };
    reader.readAsDataURL(file);
}

function clearFile() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('preview-wrap').classList.add('hidden');
    document.getElementById('dropZone').classList.remove('hidden');
    document.getElementById('analyze-button').disabled = true;
    document.getElementById('result-image').removeAttribute('src');
    resetResults();
}

async function analyze() {
    if (!selectedFile) return;
    const button = document.getElementById('analyze-button');
    button.disabled = true;
    button.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> ANALYZING';
    setStage('ingest', 'PROCESSING');
    setStage('cnn', 'PROCESSING');
    setText('analysis-title', 'Analysis in progress');
    setText('analysis-state', 'PROCESSING');
    const form = new FormData();
    form.append('file', selectedFile);
    try {
        const response = await fetch('/api/analyze', { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok || data.success === false) throw new Error(data.error || 'Analysis failed.');
        renderAnalysis(data);
    } catch (error) {
        ['map', 'lstm', 'risk', 'warning'].forEach(stage => setStage(stage, 'UNAVAILABLE'));
        setStage('cnn', 'ERROR');
        notify(error.message);
        setText('analysis-title', 'Analysis unavailable');
    } finally {
        button.disabled = false;
        button.innerHTML = '<i class="fa-solid fa-microchip"></i> ANALYZE SATELLITE IMAGE';
    }
}

function renderAnalysis(data) {
    const cnn = data.cnn || {};
    const cyclone = data.cyclone || {};
    const track = data.track || {};
    const risk = data.risk || {};
    const provenance = data.provenance || {};
    document.getElementById('result-image').src = data.image.url;
    setText('analysis-title', 'Analysis complete');
    setText('analysis-state', 'COMPLETE');
    setText('last-analysis', new Date(data.timestamp).toLocaleTimeString());
    setText('kpi-cyclone', cyclone.detected ? (cyclone.name || cyclone.id || 'DETECTED') : 'UNLINKED');
    setText('kpi-cyclone-detail', cyclone.detected ? 'Verified image mapping' : (cyclone.reason || 'No verified storm link'));
    setText('kpi-confidence', percent(cnn.confidence));
    setText('kpi-class', cnn.class_name || 'N/A');
    setText('kpi-wind', value(cyclone.wind, ' kt'));
    setText('kpi-pressure', value(cyclone.pressure, ' hPa'));
    setText('summary-cnn', 'Satellite product classified successfully.');
    setText('summary-identification', cyclone.detected ? 'Verified image-to-cyclone mapping.' : 'No verified image-to-cyclone mapping.');
    setText('summary-track', track.available ? 'LSTM forecast generated from historical sequence.' : 'Requires verified cyclone identity and historical sequence.');
    setText('summary-risk', risk.available === false ? 'Requires verified current cyclone observations.' : 'Assessment calculated from valid observations.');
    setText('cnn-prediction', (cnn.prediction || 'UNAVAILABLE').toUpperCase());
    setText('cnn-confidence', percent(cnn.confidence));
    renderProbabilities(cnn);
    setStage('ingest', 'COMPLETED');
    setStage('cnn', 'COMPLETED');
    setStage('map', cyclone.detected ? 'COMPLETED' : 'UNAVAILABLE');
    setStage('lstm', track.available ? 'COMPLETED' : 'UNAVAILABLE');
    setStage('risk', risk.available === false ? 'UNAVAILABLE' : 'COMPLETED');
    setStage('warning', risk.available === false ? 'UNAVAILABLE' : 'COMPLETED');
    document.getElementById('empty-state').classList.add('hidden');
    setText('ai-interpretation', cyclone.detected ? 'The image is linked to a verified cyclone observation. Track and risk outputs are grounded in that dataset.' : 'The CNN classified the satellite product. The image has no verified cyclone-track association, so no storm path is inferred.');
    setText('prov-cnn', provenance.cnn_source || 'N/A');
    setText('prov-mapping', provenance.satellite_mapping_source || 'N/A');
    setText('prov-ibtracs', provenance.ibtracs_source || 'N/A');
    setText('prov-confidence', provenance.mapping_confidence == null ? 'N/A' : provenance.mapping_confidence);
    setText('prov-forecast', provenance.forecast_basis || 'N/A');
    renderTrack(track);
    renderRisk(risk);
    saveHistory(data);
    renderHistory();
}

function renderProbabilities(cnn) {
    const list = document.getElementById('probability-list');
    list.innerHTML = '';
    (cnn.probabilities || []).forEach((probability, index) => {
        const name = (cnn.classes || [])[index] || `Class ${index + 1}`;
        const row = document.createElement('div');
        row.innerHTML = `<span>${escapeHtml(name)}</span><b>${percent(probability)}</b><i><em style="width:${Math.max(0, Math.min(100, probability * 100))}%"></em></i>`;
        list.appendChild(row);
    });
}

function initMap() {
    map = L.map('cyclone-map', { zoomControl: false }).setView([15, 85], 4);
    L.control.zoom({ position: 'bottomright' }).addTo(map);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { attribution: '&copy; OpenStreetMap &copy; CARTO' }).addTo(map);
}

function renderTrack(track) {
    const empty = document.getElementById('track-empty');
    const content = document.getElementById('track-content');
    setText('track-method', track.available ? 'LSTM MODEL' : 'UNAVAILABLE');
    if (!track.available) {
        empty.classList.remove('hidden');
        content.classList.add('hidden');
        setText('track-reason', track.reason || 'No verified track is available for this image.');
        return;
    }
    empty.classList.add('hidden');
    content.classList.remove('hidden');
    layers.forEach(layer => map.removeLayer(layer));
    layers = [];
    const historical = track.historical || [];
    const predicted = track.predicted || [];
    const observed = historical.map(point => [point.lat, point.lon]);
    const future = predicted.map(point => [point.lat, point.lon]);
    layers.push(L.polyline(observed, { color: '#38bdf8', weight: 3 }).addTo(map));
    layers.push(L.polyline([observed[observed.length - 1], ...future], { color: '#f59e0b', weight: 3, dashArray: '8 8' }).addTo(map));
    historical.forEach((point, index) => layers.push(L.circleMarker([point.lat, point.lon], { radius: index === historical.length - 1 ? 7 : 3, color: '#38bdf8', fillColor: '#38bdf8', fillOpacity: 1 }).addTo(map)));
    predicted.forEach(point => layers.push(L.circleMarker([point.lat, point.lon], { radius: 4, color: '#f59e0b', fillColor: '#101827', fillOpacity: 1 }).addTo(map)));
    map.fitBounds(L.latLngBounds([...observed, ...future]), { padding: [30, 30] });
    renderForecastCards(predicted);
    renderCharts(historical, predicted);
}

function renderForecastCards(points) {
    const container = document.getElementById('forecast-cards');
    container.innerHTML = '';
    points.filter(point => [3, 6, 12, 24, 36, 48].includes(Number(point.time_offset))).forEach(point => {
        const card = document.createElement('div');
        card.className = 'forecast-card';
        card.innerHTML = `<b>T+${point.time_offset}H</b><span>${Number(point.lat).toFixed(2)}N / ${Number(point.lon).toFixed(2)}E</span><small>${value(point.wind_estimated, ' kt')} | ${value(point.pressure_estimated, ' hPa')}</small>`;
        container.appendChild(card);
    });
}

function renderCharts(historical, predicted) {
    document.getElementById('charts-section').classList.remove('hidden');
    const labels = [...historical.map((_, index) => `H-${historical.length - index - 1}`), ...predicted.map(point => `+${point.time_offset}H`)];
    const bridge = values => [...historical.map(values), historical.length ? values(historical[historical.length - 1]) : null, ...predicted.map(values)];
    createChart('windChart', labels, [{ label: 'Observed', data: historical.map(point => point.wind), borderColor: '#38bdf8' }, { label: 'Predicted', data: bridge(point => point.wind_estimated), borderColor: '#f59e0b', borderDash: [6, 4] }]);
    createChart('pressureChart', labels, [{ label: 'Observed', data: historical.map(point => point.pressure), borderColor: '#38bdf8' }, { label: 'Predicted', data: bridge(point => point.pressure_estimated), borderColor: '#f59e0b', borderDash: [6, 4] }]);
    createChart('trackChart', labels, [{ label: 'Latitude', data: [...historical.map(point => point.lat), null, ...predicted.map(point => point.lat)], borderColor: '#38bdf8' }, { label: 'Longitude', data: [...historical.map(point => point.lon), null, ...predicted.map(point => point.lon)], borderColor: '#ef6c8f' }]);
}

function createChart(id, labels, datasets) {
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(document.getElementById(id), { type: 'line', data: { labels, datasets: datasets.map(data => ({ ...data, tension: .25, pointRadius: 2, spanGaps: true })) }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#a9b6c7' } } }, scales: { x: { ticks: { color: '#718197' }, grid: { color: '#1e2b3e' } }, y: { ticks: { color: '#718197' }, grid: { color: '#1e2b3e' } } } } });
}

function renderRisk(risk) {
    const available = risk.available !== false && risk.risk_score != null;
    setText('risk-chip', available ? risk.risk_level : 'UNAVAILABLE');
    setText('risk-score', available ? Number(risk.risk_score).toFixed(1) : 'N/A');
    setText('risk-level', available ? `${risk.risk_level} RISK` : 'NO ACTIVE WARNING');
    setText('kpi-risk', available ? risk.risk_level : 'N/A');
    setText('kpi-risk-score', available ? `${risk.risk_score} / 100` : 'No assessment');
    setText('risk-reason', risk.reason || 'Risk assessment unavailable.');
    setText('warning-level', available ? risk.risk_level : 'AWAITING DATA');
    setText('warning-copy', risk.reason || 'Upload an image to generate a transparent warning assessment.');
    const values = risk.factors || {};
    const labels = [['Wind speed', 'wind_score'], ['Pressure', 'pressure_score'], ['Track trend', 'trend_score'], ['Satellite context', 'satellite_score']];
    document.getElementById('factor-list').innerHTML = labels.map(([label, key]) => `<div><span>${label}</span><b>${values[key] == null ? 'N/A' : values[key] + ' / 100'}</b></div>`).join('');
    document.getElementById('actions').innerHTML = (risk.recommended_actions || ['Monitor official meteorological updates.', 'Follow local authority guidance.']).map(action => `<li>${escapeHtml(action)}</li>`).join('');
}

async function loadStatus() {
    try {
        const data = await fetch('/api/status').then(response => response.json());
        const models = data.models || {};
        setText('system-status', 'ONLINE');
        setStatus('cnn', models.cnn); setStatus('lstm', models.lstm); setStatus('risk', models.risk_engine);
        setText('quality-track', data.dataset ? 'AVAILABLE' : 'UNAVAILABLE');
    } catch { setText('system-status', 'OFFLINE'); }
}

async function loadMetrics() {
    try {
        const data = await fetch('/api/model-metrics').then(response => response.json());
        const cnn = data.cnn || {}; const lstm = data.lstm || {};
        setText('metric-accuracy', percent(cnn.accuracy)); setText('metric-precision', percent(cnn.precision)); setText('metric-recall', percent(cnn.recall)); setText('metric-f1', percent(cnn.f1_score));
        setText('metric-lat', value(lstm.latitude_mae, ' deg')); setText('metric-lon', value(lstm.longitude_mae, ' deg')); setText('metric-wind', value(lstm.wind_mae, ' kt')); setText('metric-pressure', value(lstm.pressure_mae, ' hPa'));
    } catch { /* unavailable values remain visible */ }
}

async function loadCyclones() {
    const select = document.getElementById('cycloneSelect');
    try {
        const cyclones = await fetch('/api/cyclones').then(response => response.json());
        cyclones.forEach(cyclone => { const option = document.createElement('option'); option.value = cyclone.id; option.textContent = cyclone.name || cyclone.id; select.appendChild(option); });
    } catch { select.disabled = true; }
}

function setupManualMode() {
    document.getElementById('manual-track').addEventListener('click', async () => {
        const cycloneId = document.getElementById('cycloneSelect').value;
        if (!cycloneId) return notify('Select a historical cyclone first.');
        const response = await fetch('/predict/track', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cyclone_id: cycloneId }) });
        const data = await response.json();
        if (!response.ok) return notify(data.error || 'Manual analysis failed.');
        renderTrack({ available: data.predicted.length > 0, historical: data.historical, predicted: data.predicted });
        setText('track-method', data.demo_mode ? 'BASELINE' : 'LSTM MODEL');
        setText('prov-cnn', 'N/A - manual analysis');
        setText('prov-mapping', 'N/A - manual analysis');
        setText('prov-ibtracs', 'data/processed/ibtracs_ni_processed.csv');
        setText('prov-confidence', 'N/A - manual analysis');
        setText('prov-forecast', data.demo_mode ? 'BASELINE' : 'LSTM MODEL using historical IBTrACS observations');
    });
}

function setStage(stage, state) { const element = document.querySelector(`[data-stage="${stage}"]`); if (element) { element.className = `pipeline-step ${state.toLowerCase()}`; element.querySelector('small').textContent = state; } }
function setStatus(key, ready) { setText(`${key}-status`, ready ? 'READY' : 'UNAVAILABLE'); const dot = document.getElementById(`${key}-dot`); if (dot) dot.classList.toggle('ready', !!ready); setText(`quality-${key}`, ready ? 'READY' : 'UNAVAILABLE'); }
function resetResults() { ['kpi-cyclone', 'kpi-confidence', 'kpi-wind', 'kpi-pressure', 'kpi-risk', 'cnn-confidence', 'risk-score'].forEach(id => setText(id, 'N/A')); setText('cnn-prediction', 'UNAVAILABLE'); setText('risk-level', 'NO ACTIVE WARNING'); document.getElementById('track-empty').classList.remove('hidden'); document.getElementById('track-content').classList.add('hidden'); }
function saveHistory(data) { const history = JSON.parse(localStorage.getItem('cyclone-history') || '[]'); history.unshift({ time: data.timestamp, file: data.image.filename, classification: data.cnn.class_name, confidence: data.cnn.confidence, risk: data.risk.risk_level || 'UNAVAILABLE', track: data.track.available }); localStorage.setItem('cyclone-history', JSON.stringify(history.slice(0, 5))); }
function renderHistory() { const history = JSON.parse(localStorage.getItem('cyclone-history') || '[]'); document.getElementById('history-list').innerHTML = history.length ? history.map(item => `<div><b>${escapeHtml(item.file)}</b><span>${new Date(item.time).toLocaleString()} | ${escapeHtml(item.risk)} | ${item.track ? 'TRACK' : 'NO TRACK'}</span></div>`).join('') : '<span>No completed analyses.</span>'; }
function value(number, suffix = '') { return number == null || !Number.isFinite(Number(number)) ? 'N/A' : `${Number(number).toFixed(1)}${suffix}`; }
function percent(number) { return number == null || !Number.isFinite(Number(number)) ? 'N/A' : `${(Number(number) * 100).toFixed(1)}%`; }
function formatBytes(bytes) { return `${(bytes / 1024 / 1024).toFixed(2)} MB`; }
function setText(id, text) { const element = document.getElementById(id); if (element && text != null) element.textContent = text; }
function notify(message) { window.alert(message); }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[character]); }
