import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

import config
from prediction.image_predictor import CycloneImagePredictor
from prediction.track_predictor import CycloneTrackPredictor
from prediction.risk_assessment import CycloneRiskAssessor

app = Flask(__name__)
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

session_state = {
    "current_track": [],
    "predicted_track": []
}

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

@app.route('/about')
def about():
    try:
        return render_template('about.html', disclaimer=getattr(config, 'DISCLAIMER', ''))
    except Exception as e:
        return f"Error rendering about: {str(e)}", 500

@app.route('/predict/image', methods=['POST'])
def predict_image():
    try:
        upload = request.files.get('file') or request.files.get('image')
        if upload is None:
            return jsonify({'success': False, 'error': 'No file part'}), 400
        
        file = upload
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        filename = secure_filename(file.filename)
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
                result = image_predictor.predict(filepath)
                result['image_url'] = f"/uploads/{filename}"
                result['success'] = 'error' not in result
                result['confidence_percent'] = round(result.get('confidence', 0) * 100, 1)
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
        cyclone_id = data.get('cyclone_id')
        
        if track_predictor:
            historical = track_predictor.get_historical_track(cyclone_id)
            session_state['current_track'] = historical
            
            predicted = track_predictor.predict_track(historical)
            session_state['predicted_track'] = predicted
            
            return jsonify({
                'success': True,
                'historical': historical,
                'predicted': predicted,
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
        if not session_state['current_track'] and track_predictor:
            session_state['current_track'] = track_predictor.get_historical_track()
            
        return jsonify({
            'track': session_state['current_track']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predicted-track', methods=['GET'])
def get_predicted_track():
    try:
        if not session_state['predicted_track'] and track_predictor:
            historical = session_state.get('current_track', [])
            if not historical:
                historical = track_predictor.get_historical_track()
                session_state['current_track'] = historical
            session_state['predicted_track'] = track_predictor.predict_track(historical)
            
        return jsonify({
            'track': session_state['predicted_track']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk', methods=['GET'])
def get_risk():
    try:
        if not risk_assessor:
            return jsonify({'error': 'Risk assessor not initialized'}), 500
            
        historical = session_state.get('current_track', [])
        predicted = session_state.get('predicted_track', [])
        
        if not historical and track_predictor:
            historical = track_predictor.get_historical_track()
            
        if not historical:
            return jsonify({'error': 'No tracking data available'}), 400
            
        current_pos = historical[-1]
        wind_speed = current_pos.get('wind', 0)
        pressure = current_pos.get('pressure', None)
        
        risk = risk_assessor.assess_risk(
            wind_speed=wind_speed,
            predicted_track=predicted,
            current_position=current_pos,
            pressure=pressure
        )
        
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
