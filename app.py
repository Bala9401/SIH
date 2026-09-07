import os
import json
import uuid
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
            
            if image_predictor:
                try:
                    result = image_predictor.predict(filepath, original_filename)
                except TypeError:
                    result = image_predictor.predict(filepath)
                result['image_url'] = f"/uploads/{filename}"
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
                        session['latest_track_prediction'] = {
                            'historical': historical,
                            'predicted': predicted,
                        }
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

@app.route('/predict/track', methods=['POST'])
def predict_track():
    try:
        data = request.json or {}
        cyclone_id = data.get('cyclone_id') or session.get('identified_cyclone_id')
        if not cyclone_id:
            return jsonify({'success': False, 'error': 'No identified cyclone is available. Analyze an image or choose a manual fallback cyclone.'}), 400
        
        if track_predictor:
            historical = track_predictor.get_historical_track(cyclone_id)
            if not historical:
                return jsonify({'success': False, 'error': 'Historical cyclone track unavailable.'}), 404
            predicted = track_predictor.predict_track(historical)
            session['cyclone_id'] = cyclone_id
            session['identified_cyclone_id'] = cyclone_id
            cyclone_name = track_predictor.get_cyclone_name(cyclone_id)
            session['identified_cyclone_name'] = cyclone_name
            session['identified_cyclone_match_confidence'] = None
            session['latest_track_prediction'] = {'historical': historical, 'predicted': predicted}
            
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
                            'error': 'Unable to identify cyclone from uploaded image. Select a cyclone as a manual fallback.'}), 400
            
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

@app.route('/api/demo-status', methods=['GET'])
def get_demo_status():
    try:
        status = {
            'global_demo': getattr(config, 'DEMO_MODE', True),
            'image_predictor': image_predictor.demo_mode if image_predictor else True,
            'track_predictor': track_predictor.demo_mode if track_predictor else True,
            'risk_assessor': getattr(risk_assessor, 'demo_mode', True) if risk_assessor else True
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Return the frontend-compatible aggregate runtime status."""
    return jsonify({
        'demo_mode': any((
            image_predictor is None or image_predictor.demo_mode,
            track_predictor is None or track_predictor.demo_mode,
            risk_assessor is None or risk_assessor.demo_mode,
        )),
        'models': {
            'cnn': bool(image_predictor and not image_predictor.demo_mode),
            'lstm': bool(track_predictor and not track_predictor.demo_mode),
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
