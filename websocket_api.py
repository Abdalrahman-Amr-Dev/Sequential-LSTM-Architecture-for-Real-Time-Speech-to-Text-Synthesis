import os
from flask import Flask , request  
from flask_socketio import SocketIO, emit
import numpy as np
import whisper
import threading
from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
socketio = SocketIO(app, cors_allowed_origins="*")
usedModel = os.getenv('WHISPER_MODEL', 'tiny')
# Load model once
model = whisper.load_model(usedModel) 

class RealTimeAudioProcessor:
    def __init__(self, sample_rate=16000, chunk_duration=0.5):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_buffer = deque(maxlen=int(sample_rate * 5))  # 5 second buffer
        self.lock = threading.Lock()
    
    def add_audio(self, audio_bytes):
        """Add audio bytes to buffer"""
        try:
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            with self.lock:
                self.audio_buffer.extend(audio_data)
        except Exception as e:
            print(f"Error adding audio: {e}")
    
    def get_buffered_audio(self):
        """Get current buffered audio"""
        with self.lock:
            if len(self.audio_buffer) > 0:
                return np.array(list(self.audio_buffer), dtype=np.float32)
        return None
    
    def transcribe_buffered(self):
        """Transcribe current buffer"""
        try:
            audio = self.get_buffered_audio()
            if audio is None or len(audio) < self.sample_rate:
                return None
            
            result = model.transcribe(audio, language="en", fp16=False)
            return result["text"]
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
    
    def clear_buffer(self):
        """Clear the audio buffer"""
        with self.lock:
            self.audio_buffer.clear()

# Global processor
processor = RealTimeAudioProcessor()


@socketio.on('connect')
def handle_connect(auth=None):
    print(f"Client connected: {request.sid}")
    emit('connection_response', {'data': 'Connected to real-time API'})

@socketio.on('audio_stream')
def handle_audio_stream(data):
    try:
        audio_bytes = bytes(data['audio'])
        processor.add_audio(audio_bytes)
        emit('buffer_update', {
            'buffer_size': len(processor.audio_buffer)
        }, broadcast=False)
    except Exception as e:
        print(f"Error handling audio: {e}")

@socketio.on('transcribe_request')
def handle_transcribe_request():
    try:
        text = processor.transcribe_buffered()
        emit('transcription_result', {
            'text': text if text else 'No audio to transcribe',
            'success': True
        }, broadcast=False)
    except Exception as e:
        emit('transcription_result', {
            'text': f'Error: {str(e)}',
            'success': False
        }, broadcast=False)

@socketio.on('clear_buffer')
def handle_clear_buffer():
    processor.clear_buffer()
    emit('buffer_update', {'buffer_size': 0}, broadcast=False)

if __name__ == '__main__':
    print("Starting WebSocket Real-Time Speech-to-Text API...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
