import os
import json
import hashlib
import uuid
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
from PIL import Image

import config
from prediction.image_predictor import CycloneImagePredictor
from prediction.track_predictor import CycloneTrackPredictor
from prediction.risk_assessment import CycloneRiskAssessor

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'sih-cyclone-development-key')
app.config['UPLOAD_FOLDER'] = os.path.join(getattr(config, 'BASE_DIR', os.getcwd()), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB limit
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("Initializing models...")
try:
    image_predictor = CycloneImagePredictor()
    track_predictor = CycloneTrackPredictor()
    risk_assessor = CycloneRiskAssessor()
except Exception as e:
    print(f"Error initializing models: {e}")
    image_predictor = None
    track_predictor = None
    risk_assessor = None

@app.route('/')
def index():
    try:
        return render_template('index.html', disclaimer=getattr(config, 'DISCLAIMER', ''))
    except Exception as e:
        return f"Error rendering index: {str(e)}", 500

@app.route('/dashboard')
def dashboard():
    try:
        return render_template('dashboard.html', demo_mode=getattr(config, 'DEMO_MODE', True), disclaimer=getattr(config, 'DISCLAIMER', ''))
    except Exception as e:
        return f"Error rendering dashboard: {str(e)}", 500

# Presentation-friendly deep links.  The responsive dashboard contains these
# panels; anchors keep one implementation of the live map and charts.
@app.route('/track-forecast')
def track_forecast():
    return dashboard()

@app.route('/intensity-forecast')
def intensity_forecast():
    return dashboard()

@app.route('/risk-assessment')
def risk_assessment_page():
    return dashboard()

@app.route('/satellite-analysis')
def satellite_analysis():
    return dashboard()

@app.route('/data-model-information')
def data_model_information():
    return about()

@app.route('/about')
def about():
    try:
        return render_template('about.html', disclaimer=getattr(config, 'DISCLAIMER', ''))
    except Exception as e:
        return f"Error rendering about: {str(e)}", 500

@app.route('/predict/image', methods=['POST'])
def predict_image():
    try:
        for key in ('latest_image_analysis', 'cyclone_id', 'identified_cyclone_id',
                    'identified_cyclone_name', 'identified_cyclone_match_confidence',
                    'latest_track_prediction'):
            session.pop(key, None)
        upload = request.files.get('file') or request.files.get('image')
        if upload is None:
            return jsonify({'success': False, 'error': 'No file part'}), 400
        
        file = upload
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        original_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4().hex}_{original_filename}"
        if os.path.splitext(filename)[1].lower() not in ALLOWED_IMAGE_EXTENSIONS:
            return jsonify({'success': False, 'error': 'Unsupported image format'}), 415

        try:
            image = Image.open(file)
            image.verify()
            file.seek(0)
        except Exception:
            return jsonify({'success': False, 'error': 'Uploaded file is not a readable image'}), 400
            
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            with open(filepath, 'rb') as saved_file:
                image_hash = hashlib.sha256(saved_file.read()).hexdigest()
            app.logger.info('Uploaded filename=%s sha256=%s', original_filename, image_hash)
            
            if image_predictor:
                try:
                    result = image_predictor.predict(filepath, original_filename)
                except TypeError:
                    result = image_predictor.predict(filepath)
                result['image_url'] = f"/uploads/{filename}"
                result['sha256'] = image_hash
                result['success'] = 'error' not in result
                result['confidence_percent'] = round(result.get('confidence', 0) * 100, 1)
                if result.get('success', True):
                    identification = track_predictor.identify_cyclone(result) if track_predictor else {
                        'matched': False,
                        'reason': 'Track predictor is unavailable.'
                    }
                    historical = []
                    predicted = []
                    if identification.get('matched'):
                        cyclone_id = identification['cyclone_id']
                        historical = track_predictor.get_historical_track(cyclone_id)
                        predicted = track_predictor.predict_track(historical)
                        session['cyclone_id'] = cyclone_id
                        session['identified_cyclone_id'] = cyclone_id
                        session['identified_cyclone_name'] = identification.get('cyclone_name')
                        session['identified_cyclone_match_confidence'] = identification.get('match_confidence')
                    session['latest_image_analysis'] = {
                        'available': True,
                        'prediction': result.get('prediction'),
                        'class_name': result.get('class_name'),
                        'confidence': result.get('confidence', 0),
                        'task': result.get('task', 'satellite_product_classification'),
                        'image_valid': result.get('image_valid', True),
                        'image_model_available': result.get('image_model_available', False),
                        'cyclone_detected': result.get('cyclone_detected'),
                        'cyclone_class': result.get('cyclone_class', result.get('class_name')),
                        'image_filename': result.get('image_filename'),
                        'timestamp': result.get('timestamp'),
                        'latitude': result.get('latitude'),
                        'longitude': result.get('longitude'),
                        'cyclone_name': None,
                        'cyclone_name_available': False,
                        'cyclone_detection_supported': result.get('cyclone_detection_supported', False),
                        'intensity_classification_available': result.get('intensity_classification_available', False),
                        'intensity_prediction_available': result.get('intensity_prediction_available', False),
                        'risk_prediction_from_image_available': result.get('risk_prediction_from_image_available', False),
                        'estimated_wind_speed_kt': result.get('estimated_wind_speed_kt'),
                        'estimated_pressure_hpa': result.get('estimated_pressure_hpa'),
                        'image_risk_score': result.get('image_risk_score'),
                        'image_risk_level': result.get('image_risk_level', 'UNAVAILABLE'),
                        'image_risk_basis': result.get('image_risk_basis'),
                        'all_probabilities': result.get('all_probabilities', []),
                        'image_url': result.get('image_url')
                    }
                    result['cyclone_identification'] = identification
                    result['historical_track'] = historical
                    result['predicted_track'] = predicted
                result['satellite_analysis'] = session.get('latest_image_analysis')
                return jsonify(result)
            else:
                return jsonify({'success': False, 'error': 'Image predictor not initialized'}), 500
                
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/analyze', methods=['POST'])
def analyze_uploaded_image():
    """Run image analysis and only forecast when a verified track is available."""
    for key in ('latest_image_analysis', 'cyclone_id', 'identified_cyclone_id',
                'identified_cyclone_name', 'identified_cyclone_match_confidence',
                'latest_track_prediction', 'latest_risk'):
        session.pop(key, None)

    upload = request.files.get('file') or request.files.get('image')
    if upload is None or not upload.filename:
        return jsonify({'success': False, 'error': 'Upload a satellite image to begin analysis.'}), 400

    original_filename = secure_filename(upload.filename)
    extension = os.path.splitext(original_filename)[1].lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({'success': False, 'error': 'Unsupported image format.'}), 415

    try:
        image = Image.open(upload)
        image.verify()
        upload.seek(0)
    except Exception:
        return jsonify({'success': False, 'error': 'Uploaded file is not a readable image.'}), 400

    filename = f"{uuid.uuid4().hex}_{original_filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    upload.save(filepath)

    try:
        with Image.open(filepath) as saved_image:
            width, height = saved_image.size
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as saved_file:
            for chunk in iter(lambda: saved_file.read(1024 * 1024), b''):
                sha256.update(chunk)
        image_hash = sha256.hexdigest()
        app.logger.info('Uploaded filename=%s sha256=%s', original_filename, image_hash)
        image_result = image_predictor.predict(filepath, original_filename) if image_predictor else {
            'prediction': 'Unavailable', 'error': 'CNN analysis unavailable.'
        }
        image_result['sha256'] = image_hash
        image_result['image_url'] = f'/uploads/{filename}'
        image_result['confidence_percent'] = round(image_result.get('confidence', 0) * 100, 1)
        if image_result.get('prediction') in ('Error', 'Unavailable'):
            return jsonify({'success': False, 'error': image_result.get('error', 'CNN analysis unavailable.'),
                            'image': {'url': image_result['image_url'], 'filename': original_filename,
                                      'width': width, 'height': height}}), 503

        identification = track_predictor.identify_cyclone(image_result) if track_predictor else {
            'matched': False, 'reason': 'Track predictor is unavailable.'
        }
        app.logger.info(
            'SHA-256 mapping found=%s cyclone_id=%s cyclone_name=%s',
            identification.get('matched'), identification.get('cyclone_id'), identification.get('cyclone_name')
        )
        historical = []
        predicted = []
        risk = None
        track_available = False
        track_reason = identification.get('reason')

        if identification.get('matched') and track_predictor:
            historical = track_predictor.get_track_through_observation(
                identification['cyclone_id'], identification.get('timestamp')
            )
            app.logger.info(
                'IBTrACS lookup cyclone_id=%s matched_timestamp=%s observations_through_match=%s',
                identification['cyclone_id'], identification.get('timestamp'), len(historical)
            )
            sequence_ready = len(historical) >= track_predictor.sequence_length and all(
                point.get(field) is not None
                for point in historical[-track_predictor.sequence_length:]
                for field in ('lat', 'lon', 'wind', 'pressure')
            )
            if sequence_ready and not track_predictor.demo_mode:
                predicted = track_predictor.predict_track(historical, allow_baseline=False)
                track_available = bool(predicted)
                app.logger.info('LSTM prediction cyclone_id=%s points=%s', identification['cyclone_id'], len(predicted))
                track_reason = None if track_available else 'LSTM forecast could not be generated.'
            else:
                track_reason = 'At least six complete real observations and a loaded LSTM model are required.'

            if historical and risk_assessor:
                current = historical[-1]
                if all(current.get(field) is not None for field in ('lat', 'lon', 'wind')):
                    risk = risk_assessor.assess_risk(
                        wind_speed=current.get('wind'), predicted_track=predicted,
                        current_position=current, pressure=current.get('pressure'),
                        satellite_analysis=image_result)
                    app.logger.info(
                        'Risk inputs filename=%s sha256=%s cyclone_id=%s cyclone_name=%s '
                        'current_wind=%s current_pressure=%s current_lat=%s current_lon=%s '
                        'forecast_wind=%s forecast_pressure=%s forecast_positions=%s uncertainty=%s '
                        'wind_score=%s pressure_score=%s proximity_score=%s trend_score=%s uncertainty_score=%s '
                        'final_score=%s final_level=%s',
                        original_filename, image_hash, identification.get('cyclone_id'),
                        identification.get('cyclone_name'), current.get('wind'), current.get('pressure'),
                        current.get('lat'), current.get('lon'),
                        [point.get('wind_estimated') for point in predicted],
                        [point.get('pressure_estimated') for point in predicted],
                        [(point.get('lat'), point.get('lon')) for point in predicted],
                        [point.get('uncertainty_radius_km') for point in predicted],
                        risk.get('factors', {}).get('wind_score'), risk.get('factors', {}).get('pressure_score'),
                        risk.get('factors', {}).get('proximity_score'), risk.get('factors', {}).get('trend_score'),
                        risk.get('factors', {}).get('uncertainty_score'), risk.get('risk_score'), risk.get('risk_level')
                    )

        session['latest_image_analysis'] = image_result
        if identification.get('matched'):
            session['identified_cyclone_id'] = identification.get('cyclone_id')
            session['identified_cyclone_name'] = identification.get('cyclone_name')
            session['identified_cyclone_match_confidence'] = identification.get('match_confidence')

        current = historical[-1] if historical else {}
        mapping_source = os.path.relpath(track_predictor.cyclone_matcher.mapping_path, config.BASE_DIR) if track_predictor else None
        response = {
            'success': True,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'image': {'url': image_result['image_url'], 'filename': original_filename,
                      'width': width, 'height': height, 'size_bytes': os.path.getsize(filepath)},
                'cnn': {'prediction': image_result.get('prediction'), 'class_name': image_result.get('class_name'),
                    'confidence': image_result.get('confidence'), 'probabilities': image_result.get('all_probabilities', []),
                    'classes': image_predictor.class_names if image_predictor else [], 'task': image_result.get('task'),
                    'sha256': image_result.get('sha256')},
                        'match': {'matched': identification.get('matched', False), 'method': identification.get('method'),
                                            'confidence': identification.get('match_confidence'), 'cyclone_id': identification.get('cyclone_id'),
                                            'cyclone_name': identification.get('cyclone_name'), 'timestamp': identification.get('timestamp'),
                                            'latitude': identification.get('latitude'), 'longitude': identification.get('longitude'),
                                            'wind_speed_kmh': identification.get('wind_speed_kmh'),
                                            'pressure_hpa': identification.get('pressure_hpa'), 'intensity': identification.get('intensity'),
                                            'reason': identification.get('reason')},
            'cyclone': {'detected': identification.get('matched', False), 'name': identification.get('cyclone_name'),
                        'id': identification.get('cyclone_id'), 'latitude': current.get('lat'),
                        'longitude': current.get('lon'), 'wind': current.get('wind'), 'pressure': current.get('pressure'),
                        'observation_time': current.get('time'), 'match_confidence': identification.get('match_confidence'),
                        'reason': identification.get('reason'), 'timestamp': identification.get('timestamp'),
                        'match_method': identification.get('method'), 'source': identification.get('source')},
            'track': {'available': track_available, 'historical': historical, 'predicted': predicted,
                      'forecast_hours': 48 if track_available else 0,
                      'uncertainty': [p.get('uncertainty_radius_km') for p in predicted],
                      'method': 'LSTM MODEL' if track_available else None, 'reason': track_reason},
            'current_conditions': {
                'time': current.get('time'), 'wind': current.get('wind'), 'pressure': current.get('pressure'),
                'latitude': current.get('lat'), 'longitude': current.get('lon'),
            },
            'forecast': {'points': predicted, 'horizon_hours': 48 if track_available else 0},
            'uncertainty_details': {'radii_km': [p.get('uncertainty_radius_km') for p in predicted]},
            'provenance': {
                'cnn_source': 'models/cyclone_cnn.keras; models/metadata.json',
                'satellite_mapping_source': mapping_source,
                'ibtracs_source': 'data/processed/ibtracs_ni_processed.csv' if identification.get('matched') else None,
                'mapping_confidence': identification.get('match_confidence'),
                'forecast_basis': 'LSTM MODEL using verified historical observations' if track_available else 'N/A for uploaded image',
            },
            'risk': risk if risk else {'available': False, 'risk_level': 'UNAVAILABLE', 'risk_score': None,
                                       'reason': 'Risk assessment requires a verified cyclone observation.'}
        }
        response.update({
            'image_analysis': response['cnn'],
            'cyclone_identification': response['cyclone'],
            'ibtracs_match': {
                'available': bool(identification.get('matched')),
                'cyclone_id': identification.get('cyclone_id'),
                'source': response['provenance']['ibtracs_source'],
                'reason': identification.get('reason'),
            },
            'historical_track': historical,
            'lstm_prediction': predicted,
            'uncertainty': response['track']['uncertainty'],
            'risk_assessment': response['risk'],
        })
        return jsonify(response)
    except Exception as error:
        return jsonify({'success': False, 'error': str(error)}), 500

@app.route('/predict/track', methods=['POST'])
def predict_track():
    """Reject direct storm selection; forecasts must originate from an image."""
    data = request.json or {}
    if data.get('cyclone_id'):
        return jsonify({'success': False, 'error': 'Cyclone identity is determined automatically from the uploaded image.'}), 400
    if not session.get('identified_cyclone_id'):
        return jsonify({'success': False, 'error': 'Upload a satellite image to identify a cyclone before forecasting.'}), 400
    try:
        cyclone_id = session['identified_cyclone_id']
        
        if track_predictor:
            historical = track_predictor.get_historical_track(cyclone_id)
            if not historical:
                return jsonify({'success': False, 'error': 'Historical cyclone track unavailable.'}), 404
            predicted = track_predictor.predict_track(historical, allow_baseline=False)
            session['cyclone_id'] = cyclone_id
            session['identified_cyclone_id'] = cyclone_id
            cyclone_name = track_predictor.get_cyclone_name(cyclone_id)
            session['identified_cyclone_name'] = cyclone_name
            session['identified_cyclone_match_confidence'] = None
            
            return jsonify({
                'success': True,
                'historical': historical,
                'predicted': predicted,
                'cyclone_id': cyclone_id,
                'cyclone_name': cyclone_name,
                'demo_mode': track_predictor.demo_mode,
                'forecast_horizon_hours': 48,
                'forecast_step_hours': 3,
                'uncertainty_label': 'Model-derived prototype uncertainty corridor based on held-out errors.'
            })
        else:
            return jsonify({'error': 'Track predictor not initialized'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cyclones', methods=['GET'])
def get_cyclones():
    try:
        if track_predictor:
            return jsonify(track_predictor.get_available_cyclones())
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/current-cyclone', methods=['GET'])
def get_current_cyclone():
    try:
        historical = track_predictor.get_historical_track(session.get('cyclone_id')) if track_predictor else []
        return jsonify({
            'track': historical
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predicted-track', methods=['GET'])
def get_predicted_track():
    try:
        historical = track_predictor.get_historical_track(session.get('cyclone_id')) if track_predictor else []
        predicted = track_predictor.predict_track(historical) if track_predictor else []
        return jsonify({
            'track': predicted
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk', methods=['GET'])
def get_risk():
    try:
        if not risk_assessor:
            return jsonify({'error': 'Risk assessor not initialized'}), 500

        cyclone_id = session.get('identified_cyclone_id') or session.get('cyclone_id')
        if not cyclone_id:
            return jsonify({'success': False, 'risk_level': 'INSUFFICIENT DATA',
                            'error': 'Unable to identify a cyclone from the uploaded image.'}), 400
            
        historical = track_predictor.get_historical_track(cyclone_id) if track_predictor else []
        predicted = track_predictor.predict_track(historical) if track_predictor else []
            
        if not historical:
            return jsonify({'success': False, 'risk_level': 'INSUFFICIENT DATA',
                            'error': 'No tracking data available for risk calculation.'}), 400
            
        current_pos = historical[-1]
        wind_speed = current_pos.get('wind')
        pressure = current_pos.get('pressure', None)
        satellite_analysis = session.get('latest_image_analysis')
        
        risk = risk_assessor.assess_risk(
            wind_speed=wind_speed,
            predicted_track=predicted,
            current_position=current_pos,
            pressure=pressure,
            satellite_analysis=satellite_analysis
        )
        risk['identified_cyclone'] = {
            'cyclone_id': cyclone_id,
            'cyclone_name': session.get('identified_cyclone_name') or (track_predictor.get_cyclone_name(cyclone_id) if track_predictor else None),
            'match_confidence': session.get('identified_cyclone_match_confidence'),
        }
        if risk['identified_cyclone']['cyclone_name']:
            risk['reason'] = f"{risk['identified_cyclone']['cyclone_name']}: {risk.get('reason', '')}"
        
        return jsonify(risk)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-metrics', methods=['GET'])
def get_model_metrics():
    try:
        metrics_dir = os.path.join(getattr(config, 'RESULTS_DIR', ''), 'metrics')
        metrics = {'available': False}
        for name, key in [('cnn_metrics.json', 'cnn'), ('lstm_metrics.json', 'lstm')]:
            path = os.path.join(metrics_dir, name)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    metrics[key] = json.load(f)
                metrics['available'] = True
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Return the frontend-compatible aggregate runtime status."""
    dataset_available = os.path.exists(os.path.join(config.DATA_DIR, 'processed', 'cyclone_tracks.json'))
    return jsonify({
        'demo_mode': any((
            image_predictor is None or image_predictor.demo_mode,
            track_predictor is None or track_predictor.demo_mode,
            risk_assessor is None or risk_assessor.demo_mode,
        )),
        'models': {
            'cnn': bool(image_predictor and not image_predictor.demo_mode),
            'lstm': bool(track_predictor and not track_predictor.demo_mode),
            'risk_engine': risk_assessor is not None,
        },
        'dataset': dataset_available,
        'components': {
            'cnn': 'READY' if image_predictor and not image_predictor.demo_mode else 'UNAVAILABLE',
            'lstm': 'READY' if track_predictor and not track_predictor.demo_mode else 'UNAVAILABLE',
            'risk_engine': 'READY' if risk_assessor else 'UNAVAILABLE',
            'track_dataset': 'AVAILABLE' if dataset_available else 'UNAVAILABLE',
        }
    })

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
