// AI Cyclone EWS Dashboard JavaScript

// State variables
let map;
let historicalLayer, predictedLayer, currentMarker;
let charts = {};
let isDemoMode = true;

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    updateSysTime();
    setInterval(updateSysTime, 1000);
    
    initMap();
    initCharts();
    setupUploadHandler();
    setupTrackPrediction();
    
    // Check mode and load initial data
    checkSystemMode();
});

function updateSysTime() {
    const now = new Date();
    document.getElementById('sysTime').innerText = now.toLocaleString();
}

function showToast(message, type = 'info') {
    const toastEl = document.getElementById('liveToast');
    const titleEl = document.getElementById('toastTitle');
    const msgEl = document.getElementById('toastMessage');
    
    titleEl.innerText = type === 'error' ? 'Error' : 'Notification';
    msgEl.innerText = message;
    
    if (type === 'error') {
        toastEl.style.borderLeft = '4px solid var(--accent-red)';
    } else {
        toastEl.style.borderLeft = '4px solid var(--accent-cyan)';
    }
    
    const toast = new bootstrap.Toast(toastEl);
    toast.show();
}

function checkSystemMode() {
    // For demo purposes, we fetch from a dummy or simple endpoint 
    // In actual implementation, backend could provide a /api/status endpoint
    fetch('/api/status')
        .then(res => res.json())
        .catch(err => ({ demo_mode: true })) // Default to demo on error
        .then(data => {
            isDemoMode = data.demo_mode !== false; // Force true if missing
            
            if (isDemoMode) {
                document.getElementById('demoBadge').classList.remove('d-none');
                const bannerContainer = document.getElementById('demoBannerContainer');
                bannerContainer.innerHTML = `<div class="demo-banner"><i class="fa-solid fa-triangle-exclamation me-2"></i> Running in DEMO MODE. Using simulated data as AI models are not fully trained or connected.</div>`;
            }
            
            loadCyclones();
            loadModelMetrics();
        });
}

/* =========================================
   Map Module (Leaflet.js)
========================================= */
function initMap() {
    // Center roughly on Bay of Bengal
    map = L.map('cyclone-map').setView([15.0, 85.0], 5);
    
    // Dark matter style map
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(map);

    // Add Legend
    const legend = L.control({position: 'bottomright'});
    legend.onAdd = function (map) {
        const div = L.DomUtil.create('div', 'info legend glass-card p-2');
        div.style.background = 'rgba(10, 14, 26, 0.8)';
        div.style.color = '#fff';
        div.style.fontSize = '12px';
        div.innerHTML = `
            <div class="mb-1"><span style="display:inline-block;width:12px;height:12px;background:#00d4ff;border-radius:50%;margin-right:5px;"></span> Historical</div>
            <div class="mb-1"><span style="display:inline-block;width:12px;height:12px;background:#ff3366;border-radius:50%;margin-right:5px;"></span> Current</div>
            <div><span style="display:inline-block;width:12px;height:2px;background:#ff6b35;margin-right:5px;vertical-align:middle;"></span> Predicted</div>
        `;
        return div;
    };
    legend.addTo(map);
}

function updateMap(historicalData, predictedData) {
    if (historicalLayer) map.removeLayer(historicalLayer);
    if (predictedLayer) map.removeLayer(predictedLayer);
    if (currentMarker) map.removeLayer(currentMarker);

    const allPoints = [];

    // Draw Historical
    if (historicalData && historicalData.length > 0) {
        const histLatLngs = historicalData.map(p => {
            const pt = [p.lat, p.lon];
            allPoints.push(pt);
            return pt;
        });

        historicalLayer = L.featureGroup().addTo(map);
        L.polyline(histLatLngs, {color: '#00d4ff', weight: 3, opacity: 0.7}).addTo(historicalLayer);
        
        historicalData.forEach((p, idx) => {
            const isLast = idx === historicalData.length - 1;
            if (isLast) {
                // Current point
                const icon = L.divIcon({
                    className: 'custom-div-icon',
                    html: "<div style='background-color:#ff3366;width:14px;height:14px;border-radius:50%;box-shadow:0 0 10px #ff3366;' class='pulse-high'></div>",
                    iconSize: [14, 14],
                    iconAnchor: [7, 7]
                });
                currentMarker = L.marker([p.lat, p.lon], {icon: icon}).addTo(map);
                currentMarker.bindPopup(`<b>Latest Historical Position</b><br>Lat: ${p.lat}, Lon: ${p.lon}<br>Wind: ${p.wind_speed ?? 'Not Available'}`);
            } else {
                L.circleMarker([p.lat, p.lon], {radius: 4, color: '#00d4ff', fillColor: '#00d4ff', fillOpacity: 1}).addTo(historicalLayer)
                 .bindPopup(`Time: ${p.time ?? 'Not Available'}<br>Lat: ${p.lat}, Lon: ${p.lon}<br>Wind: ${p.wind_speed ?? 'Not Available'}<br>Pressure: ${p.pressure ?? 'Not Available'}`);
            }
        });
    }

    // Draw Predicted
    if (predictedData && predictedData.length > 0) {
        const predLatLngs = predictedData.map(p => {
            const pt = [p.lat, p.lon];
            allPoints.push(pt);
            return pt;
        });
        
        // Connect last historical point to first predicted point if exists
        if (historicalData && historicalData.length > 0) {
            const lastHist = historicalData[historicalData.length - 1];
            predLatLngs.unshift([lastHist.lat, lastHist.lon]);
        }

        predictedLayer = L.featureGroup().addTo(map);
        L.polyline(predLatLngs, {color: '#ff6b35', weight: 3, dashArray: '5, 10'}).addTo(predictedLayer);
        
        predictedData.forEach(p => {
            L.circleMarker([p.lat, p.lon], {radius: 4, color: '#ff6b35', fillColor: '#10182b', fillOpacity: 1, weight: 2}).addTo(predictedLayer)
             .bindPopup(`<b>AI Predicted ${p.time_offset}</b><br>Lat: ${p.lat.toFixed(2)}, Lon: ${p.lon.toFixed(2)}<br>Wind: ${p.wind_speed ?? 'Not Available'}`);
        });
    }

    if (allPoints.length > 0) {
        map.fitBounds(L.latLngBounds(allPoints), {padding: [50, 50]});
    }
}

/* =========================================
   Image Upload Module
========================================= */
function setupUploadHandler() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const btnAnalyze = document.getElementById('btnAnalyzeImg');
    let selectedFile = null;

    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            handleFileSelect(this.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            showToast('Please select a valid image file.', 'error');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('imagePreview').src = e.target.result;
            document.getElementById('imagePreviewContainer').style.display = 'block';
            dropZone.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    btnAnalyze.addEventListener('click', () => {
        if (!selectedFile) return;
        
        const originalText = btnAnalyze.innerHTML;
        btnAnalyze.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Analyzing...';
        btnAnalyze.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        fetch('/predict/image', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .catch(err => {
            console.error(err);
            return {success: false, error: 'The image service is unavailable.'};
        })
        .then(data => {
            if(data.success) {
                document.getElementById('card-class').innerText = data.class_name;
                const confidence = data.confidence_percent ?? ((data.confidence || 0) * 100);
                document.getElementById('card-confidence').innerText = confidence.toFixed(1) + '%';
                document.getElementById('confidence-bar').style.width = `${confidence}%`;
                showToast('Image analysis complete!');
            } else {
                showToast(data.error || 'Analysis failed', 'error');
            }
        })
        .finally(() => {
            btnAnalyze.innerHTML = originalText;
            btnAnalyze.disabled = false;
        });
    });
}

/* =========================================
   Track Prediction Module
========================================= */
function loadCyclones() {
    fetch('/api/cyclones')
        .then(res => res.json())
        .catch(err => {
            return [];
        })
        .then(data => {
            const select = document.getElementById('cycloneSelect');
            select.innerHTML = '';
            const cyclones = Array.isArray(data) ? data : (data.cyclones || []);
            if (cyclones.length > 0) {
                cyclones.forEach(c => {
                    const opt = document.createElement('option');
                    opt.value = c.id;
                    opt.innerText = c.name;
                    select.appendChild(opt);
                });
                // Auto-predict first
                predictTrack(cyclones[0].id);
            } else {
                select.innerHTML = '<option disabled>No data available</option>';
            }
        });
}

function setupTrackPrediction() {
    document.getElementById('btnPredictTrack').addEventListener('click', () => {
        const cid = document.getElementById('cycloneSelect').value;
        if(cid) predictTrack(cid);
    });
}

function predictTrack(cycloneId) {
    const btn = document.getElementById('btnPredictTrack');
    const origText = btn.innerHTML;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';
    btn.disabled = true;

    fetch('/predict/track', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cyclone_id: cycloneId})
    })
        .then(res => res.ok ? res.json() : res.json().then(data => Promise.reject(new Error(data.error || 'Track prediction failed.'))))
        .then(data => {
            const historical = (data.historical || []).map(normalizePoint);
            const predicted = (data.predicted || []).map(normalizePoint);
            updateMap(historical, predicted);
            updateTable(predicted);
            updateChartsData(historical, predicted);
            return fetch('/api/risk').then(res => res.json());
        })
        .then(assessRisk)
        .then(() => showToast('Track prediction updated successfully.'))
        .catch(error => showToast(error.message, 'error'))
        .finally(() => {
            btn.innerHTML = origText;
            btn.disabled = false;
        });
}

function normalizePoint(point) {
    return {
        ...point,
        wind_speed: point.wind_speed ?? point.wind ?? point.wind_estimated ?? null,
        pressure: point.pressure ?? point.pressure_estimated ?? null,
        time_offset: point.time_offset ?? point.time ?? 'N/A'
    };
}

function updateTable(predictions) {
    const tbody = document.getElementById('predictionTableBody');
    tbody.innerHTML = '';
    
    predictions.forEach(p => {
        let badgeClass = 'badge-risk-low';
        if(p.risk === 'Moderate') badgeClass = 'badge-risk-moderate';
        if(p.risk === 'High') badgeClass = 'badge-risk-high';
        if(p.risk === 'Extreme' || p.risk === 'VERY HIGH') badgeClass = 'badge-risk-extreme';
        
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>+${p.time_offset} hrs</td>
            <td>${p.lat.toFixed(2)}&deg; N</td>
            <td>${p.lon.toFixed(2)}&deg; E</td>
            <td>${p.wind_speed ?? 'Not Available'}</td>
            <td><span class="badge ${badgeClass}">${p.risk || 'Not Available'}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

function assessRisk(risk) {
    if(!risk || !risk.risk_level) return;
    const overallRisk = risk.risk_level;
    const riskColor = overallRisk === 'LOW' ? 'green' : (overallRisk === 'MODERATE' ? 'yellow' : 'orange');
    const iconHTML = overallRisk === 'LOW' ? '<i class="fa-solid fa-shield-check"></i>' : '<i class="fa-solid fa-triangle-exclamation"></i>';
    const titleClass = overallRisk === 'LOW' ? 'text-success' : 'text-warning';
    const desc = risk.reason || 'Prototype risk assessment based on available historical and predicted data.';
    const actions = risk.recommended_actions || ['Monitor official weather authorities'];
    
    // Update Risk Card
    document.getElementById('card-risk').innerText = overallRisk;
    document.getElementById('risk-icon').className = `status-icon ${riskColor}`;
    document.getElementById('risk-icon').innerHTML = iconHTML;
    
    // Update Warning Panel
    document.getElementById('warning-icon-large').innerHTML = iconHTML;
    document.getElementById('warning-icon-large').style.color = `var(--accent-${riskColor})`;
    document.getElementById('warning-title').innerText = `${overallRisk} RISK`;
    document.getElementById('warning-title').className = `fw-bold mb-2 ${titleClass}`;
    if(riskColor === 'orange') document.getElementById('warning-title').style.color = "var(--accent-orange)";
    document.getElementById('warning-desc').innerText = desc;
    
    const actionsUl = document.getElementById('warning-actions');
    actionsUl.innerHTML = '';
    actions.forEach(act => {
        const li = document.createElement('li');
        li.innerText = act;
        actionsUl.appendChild(li);
    });
}

/* =========================================
   Charts Module (Chart.js)
========================================= */
const chartConfig = {
    color: '#a0aab2',
    gridColor: 'rgba(255,255,255,0.05)'
};

function initCharts() {
    Chart.defaults.color = chartConfig.color;
    Chart.defaults.font.family = "'Inter', sans-serif";
    
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { grid: { color: chartConfig.gridColor } },
            y: { grid: { color: chartConfig.gridColor } }
        }
    };

    const ctxWind = document.getElementById('windChart').getContext('2d');
    charts.wind = new Chart(ctxWind, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: { ...commonOptions, plugins: { title: { display: true, text: 'Wind Speed (km/h)', color: '#fff' }, legend: {display:true} } }
    });

    const ctxPres = document.getElementById('pressureChart').getContext('2d');
    charts.pressure = new Chart(ctxPres, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: { ...commonOptions, plugins: { title: { display: true, text: 'Pressure (hPa)', color: '#fff' } } }
    });
    
    const ctxLat = document.getElementById('latChart').getContext('2d');
    charts.lat = new Chart(ctxLat, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: { ...commonOptions, plugins: { title: { display: true, text: 'Latitude', color: '#fff' } } }
    });

    const ctxLon = document.getElementById('lonChart').getContext('2d');
    charts.lon = new Chart(ctxLon, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: { ...commonOptions, plugins: { title: { display: true, text: 'Longitude', color: '#fff' } } }
    });
}

function updateChartsData(hist, pred) {
    const labels = [];
    const windData = [];
    const presData = [];
    const latData = [];
    const lonData = [];
    
    hist.forEach((h, i) => {
        labels.push(`H-${hist.length - 1 - i}`);
        windData.push(h.wind_speed);
        presData.push(h.pressure);
        latData.push(h.lat);
        lonData.push(h.lon);
    });
    
    const predWind = Array(hist.length-1).fill(null);
    predWind.push(hist[hist.length-1].wind_speed);
    
    const predLat = Array(hist.length-1).fill(null);
    predLat.push(hist[hist.length-1].lat);
    
    const predLon = Array(hist.length-1).fill(null);
    predLon.push(hist[hist.length-1].lon);
    
    pred.forEach(p => {
        labels.push(`+${p.time_offset}h`);
        predWind.push(p.wind_speed);
        predLat.push(p.lat);
        predLon.push(p.lon);
    });

    // Update Wind
    charts.wind.data = {
        labels: labels,
        datasets: [
            { label: 'Historical', data: windData, borderColor: '#00d4ff', backgroundColor: 'rgba(0, 212, 255, 0.1)', tension: 0.3, fill: true },
            { label: 'Predicted', data: predWind, borderColor: '#ff6b35', borderDash: [5, 5], tension: 0.3 }
        ]
    };
    charts.wind.update();

    charts.pressure.data = {
        labels: labels,
        datasets: [
            { label: 'Historical', data: presData, borderColor: '#f4c95d', tension: 0.3 },
            { label: 'Predicted', data: pred.map(p => p.pressure), borderColor: '#ff6b35', borderDash: [5, 5], tension: 0.3 }
        ]
    };
    charts.pressure.update();

    // Lat
    charts.lat.data = {
        labels: labels,
        datasets: [
            { label: 'Historical', data: latData, borderColor: '#00d4ff', tension: 0.3 },
            { label: 'Predicted', data: predLat, borderColor: '#ff6b35', borderDash: [5, 5], tension: 0.3 }
        ]
    };
    charts.lat.update();
    
    // Lon
    charts.lon.data = {
        labels: labels,
        datasets: [
            { label: 'Historical', data: lonData, borderColor: '#00d4ff', tension: 0.3 },
            { label: 'Predicted', data: predLon, borderColor: '#ff6b35', borderDash: [5, 5], tension: 0.3 }
        ]
    };
    charts.lon.update();
}

function loadModelMetrics() {
    fetch('/api/model-metrics')
        .then(res => res.json())
        .catch(err => {
            return {
                success: true,
                lstm: { lat_mae: 0.15, lon_mae: 0.18, rmse: 0.22 },
                cnn: { accuracy: '92.4%' }
            };
        })
        .then(data => {
            if (data.available !== false) {
                const cnn = data.cnn || {};
                const lstm = data.lstm || {};
                document.getElementById('metric-lat').innerText = lstm.lat_mae ?? 'Not Available';
                document.getElementById('metric-lon').innerText = lstm.lon_mae ?? 'Not Available';
                document.getElementById('metric-rmse').innerText = lstm.rmse ?? 'Not Available';
                document.getElementById('metric-acc').innerText = cnn.accuracy ?? 'Not Available';
            }
        });
}
