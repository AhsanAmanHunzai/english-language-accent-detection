from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import json
import time
from function.compute import analyze_audio, set_session_id, ask_about_video, get_timestamped_transcript

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
UPLOAD_FOLDER = 'temp_uploads'
STATUS_FOLDER = 'status'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'mp4', 'avi', 'mov', 'webm'}

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, STATUS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_session_id():
    """Generate unique session ID for tracking processing status"""
    import uuid
    return str(uuid.uuid4())

def write_status(session_id, message, status_type='info'):
    """Write status update to file"""
    try:
        status_file = os.path.join(STATUS_FOLDER, f"{session_id}.json")
        status_data = {
            'message': message,
            'type': status_type,
            'timestamp': time.time()
        }
        
        # Read existing status or create new
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                existing_data = json.load(f)
            existing_data['logs'].append(status_data)
        else:
            existing_data = {
                'session_id': session_id,
                'status': 'processing',
                'logs': [status_data]
            }
        
        # Write updated status
        with open(status_file, 'w') as f:
            json.dump(existing_data, f)
            
    except Exception as e:
        print(f"Error writing status: {e}")

def read_status(session_id):
    """Read status from file"""
    try:
        status_file = os.path.join(STATUS_FOLDER, f"{session_id}.json")
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error reading status: {e}")
        return None

def complete_status(session_id, result=None):
    """Mark analysis as completed"""
    try:
        status_file = os.path.join(STATUS_FOLDER, f"{session_id}.json")
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status_data = json.load(f)
            status_data['status'] = 'completed'
            if result:
                status_data['result'] = result
            status_data['completed_at'] = time.time()
            
            with open(status_file, 'w') as f:
                json.dump(status_data, f)
    except Exception as e:
        print(f"Error completing status: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('index.html', page_title='About Us')

@app.route('/create-session', methods=['POST'])
def create_session():
    """Create a new session ID for tracking processing status"""
    session_id = create_session_id()
    return jsonify({'session_id': session_id})

@app.route('/status/<session_id>')
def get_status(session_id):
    """Simple polling endpoint to get current status"""
    try:
        status_data = read_status(session_id)
        if status_data:
            return jsonify(status_data)
        else:
            return jsonify({
                'session_id': session_id,
                'status': 'not_found',
                'logs': []
            }), 404
    except Exception as e:
        return jsonify({
            'error': f'Failed to get status: {str(e)}'
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if request contains JSON data (URL string)
        if request.is_json:
            data = request.get_json()
            if 'link' in data:
                link = data['link']
                session_id = data.get('session_id')
                
                if not link or not isinstance(link, str):
                    return jsonify({'error': 'Invalid link provided'}), 400
                
                if not session_id:
                    return jsonify({'error': 'Session ID required'}), 400
                
                # Set session ID for direct file-based status updates
                set_session_id(session_id)
                
                # Process URL string with session ID
                result = analyze_audio(audio_source=link, source_type='url', session_id=session_id)
                
                # Mark as completed
                complete_status(session_id, result)
                
                return jsonify(result)
            else:
                return jsonify({'error': 'Missing link parameter'}), 400
        
        # Check if request contains file upload
        elif 'file' in request.files:
            file = request.files['file']
            session_id = request.form.get('session_id')
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
                
            if not session_id:
                return jsonify({'error': 'Session ID required'}), 400
            
            if file and allowed_file(file.filename):
                # Set session ID for direct file-based status updates
                set_session_id(session_id)
                
                # Save uploaded file temporarily
                filename = file.filename
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                
                # Process uploaded file with session ID
                result = analyze_audio(audio_source=filepath, source_type='file', session_id=session_id)
                
                # Mark as completed
                complete_status(session_id, result)
                
                # Clean up temporary file
                try:
                    os.remove(filepath)
                except:
                    pass  # Don't fail if cleanup fails
                
                return jsonify(result)
            else:
                return jsonify({'error': 'Invalid file type. Supported: WAV, MP3, OGG, MP4, AVI, MOV, WEBM'}), 400
        
        else:
            return jsonify({'error': 'No data provided. Send either JSON with "link" or form-data with "file"'}), 400
    
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/ask-question', methods=['POST'])
def ask_question():
    """Handle Q&A requests about analyzed videos"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        session_id = data.get('session_id')
        question = data.get('question')
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        if not question or not question.strip():
            return jsonify({'error': 'Question is required'}), 400
        
        # Check if timestamped transcript is ready
        transcript_status = get_timestamped_transcript(session_id)
        
        if transcript_status['status'] == 'not_ready':
            return jsonify({
                'status': 'error',
                'error': 'Timestamped transcript is still being processed. Please wait a moment and try again.'
            })
        
        if transcript_status['status'] == 'error':
            return jsonify({
                'status': 'error', 
                'error': transcript_status.get('message', 'Error retrieving transcript')
            })
        
        # Ask the question using GPT-4 and timestamped transcript
        answer = ask_about_video(session_id, question.strip())
        
        return jsonify({
            'status': 'success',
            'answer': answer,
            'transcript_segments': transcript_status.get('segments_count', 0),
            'video_duration': transcript_status.get('duration', 0)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'Failed to process question: {str(e)}'
        }), 500

@app.route('/transcript-status/<session_id>', methods=['GET'])
def check_transcript_status(session_id):
    """Check if timestamped transcript is ready for a session"""
    try:
        transcript_status = get_timestamped_transcript(session_id)
        return jsonify(transcript_status)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error checking transcript status: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 