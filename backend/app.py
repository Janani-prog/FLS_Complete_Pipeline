from flask import Flask, request, jsonify, send_file, send_from_directory, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import tempfile
import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import ball_mill_analyzer as analyzer
from datetime import datetime, timedelta
import secrets

app = Flask(__name__, static_folder='../frontend', static_url_path='')
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
CORS(app, supports_credentials=True)

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USERS_FILE = Path('users.json')
HISTORY_FILE = Path('history.json')

# ... [KEEP AUTH AND USER LOGIC AS IS] ...
# (Load users_db, save_data, login/signup routes - No changes needed here)

def load_data(path):
    if not path.exists(): return {}
    with open(path, 'r') as f:
        try: return json.load(f)
        except: return {}

def save_data(path, data):
    with open(path, 'w') as f: json.dump(data, f, indent=4)

users_db = load_data(USERS_FILE)
analysis_history = load_data(HISTORY_FILE)

if 'demo@example.com' not in users_db:
    users_db['demo@example.com'] = {
        'email': 'demo@example.com',
        'password': generate_password_hash('demo123'),
        'name': 'Demo User',
        'created_at': datetime.now().isoformat()
    }
    save_data(USERS_FILE, users_db)

def login_required():
    if 'user_email' not in session:
        return jsonify({'error': 'Authentication required', 'auth_required': True}), 401
    return None

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    user = users_db.get(data.get('email', '').lower())
    if not user or not check_password_hash(user['password'], data.get('password')):
        return jsonify({'error': 'Invalid credentials'}), 401
    session['user_email'] = user['email']
    session.permanent = True
    return jsonify({'user': {'email': user['email'], 'name': user['name']}}), 200

@app.route('/api/auth/me', methods=['GET'])
def me():
    auth_err = login_required()
    if auth_err: return auth_err
    user = users_db.get(session['user_email'])
    return jsonify({'user': {'email': user['email'], 'name': user['name']}}), 200

@app.route('/api/analyze', methods=['POST'])
def analyze():
    auth_err = login_required()
    if auth_err: return auth_err
    
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
        
    try:
        points = analyzer.load_point_cloud(tmp_path)
        points, _ = analyzer.correct_yz_alignment(points)
        res = analyzer.run_analysis_logic(points)
        
        # USE ABSOLUTE OUTPUT PATH
        img_filename = f"{Path(file.filename).stem}_analysis.png"
        img_path = OUTPUT_DIR / img_filename
        
        analyzer.make_plots(points, res, img_path, file.filename)
        
        response_data = {}
        for k, v in res.items():
            if isinstance(v, (np.float64, np.float32, float)):
                response_data[k] = float(v)
            elif isinstance(v, (np.int64, np.int32, int)):
                response_data[k] = int(v)
            else:
                response_data[k] = v
        
        response_data['image_filename'] = img_filename
        response_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        response_data['filename'] = file.filename
        
        return jsonify(response_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

@app.route('/api/download-image/<filename>')
def download(filename):
    # USE ABSOLUTE PATH TO SERVE FILE
    try:
        return send_file(OUTPUT_DIR / filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'ok': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)