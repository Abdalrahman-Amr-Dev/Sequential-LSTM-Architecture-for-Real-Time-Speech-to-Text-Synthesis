# Real-Time Speech-to-Text API

A real-time socket-based API for speech-to-text transformation using OpenAI's Whisper model. Provides both raw socket and WebSocket implementations for real-time audio streaming and transcription.

## Features

✅ **Real-Time Audio Streaming** - Stream audio data continuously  
✅ **Buffering System** - Automatic audio buffering (5-second default)  
✅ **Multi-Client Support** - Handle multiple simultaneous connections  
✅ **Two API Options** - Raw Socket or WebSocket (recommended)  
✅ **JSON Protocol** - Simple, structured message format  
✅ **Web Dashboard** - Browser-based testing interface (WebSocket only)

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## API Options

### Option 1: WebSocket API (Recommended)

**Best for:** Web clients, browsers, real-time updates

```bash
python websocket_api.py
```

- Access dashboard: `http://localhost:5000`
- Supports browser audio recording
- Automatic reconnection
- Real-time transcription updates

**WebSocket Events:**

- `audio_stream` - Send audio data
- `transcribe_request` - Request transcription
- `clear_buffer` - Clear audio buffer
- `buffer_update` - Server broadcasts buffer size
- `transcription_result` - Server sends transcription

### Option 2: Raw Socket API

**Best for:** Native applications, lower overhead

```bash
python socket_api.py
```

**Commands (JSON):**

1. **Send Audio:**

```json
{
  "command": "audio",
  "data": "base64_encoded_audio"
}
```

2. **Request Transcription:**

```json
{
  "command": "transcribe"
}
```

3. **Clear Buffer:**

```json
{
  "command": "clear"
}
```

4. **Get Status:**

```json
{
  "command": "status"
}
```

## Usage Examples

### Python Socket Client

```python
from socket_client import SocketClient
import numpy as np

# Connect to server
client = SocketClient(host='localhost', port=5000)
client.connect()

# Send audio data
audio_data = np.random.randn(16000).astype(np.float32)  # 1 second @ 16kHz
response = client.send_audio(audio_data)

# Request transcription
result = client.transcribe()
print(result['text'])

# Cleanup
client.close()
```

### JavaScript WebSocket Client

```javascript
const socket = io("http://localhost:5000");

socket.on("connect", function () {
  console.log("Connected!");
});

// Send audio
socket.emit("audio_stream", {
  audio: arrayBuffer,
});

// Request transcription
socket.emit("transcribe_request", {});

// Listen for results
socket.on("transcription_result", function (data) {
  console.log("Transcribed:", data.text);
});
```

### cURL Socket Client Example

```bash
# Start server in one terminal
python socket_api.py

# Test in another terminal (using netcat)
echo '{"command": "status"}' | nc localhost 5000
```

## Architecture

```
┌─────────────────┐
│  Audio Client   │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Socket  │ (JSON protocol)
    │ Server  │
    └────┬────┘
         │
    ┌────▼─────────────────┐
    │ RealTimeAudioProcessor│
    │ - Audio Buffer       │
    │ - Threading Safety   │
    └────┬─────────────────┘
         │
    ┌────▼─────────┐
    │ Whisper Model│
    │ Transcription│
    └──────────────┘
```

## Configuration

Edit the server files to change:

```python
# Host and port
SocketServer(host='0.0.0.0', port=5000)

# Audio buffer size (seconds)
RealTimeAudioProcessor(chunk_duration=0.5)
```

## Performance Tips

1. **Reduce Model Size** - Use `tiny` or `small` for faster inference:

   ```python
   model = whisper.load_model("tiny")  # Fastest
   ```

2. **Stream Processing** - Send audio in chunks rather than all at once

3. **Buffer Management** - Transcribe periodically to avoid memory buildup

4. **Threading** - Both servers use threading for concurrent clients

## File Structure

```
.
├── socket_api.py           # Raw socket server
├── websocket_api.py        # WebSocket + Flask dashboard
├── socket_client.py        # Python socket client example
├── main.py                 # Original Whisper implementation
└── requirements.txt        # Dependencies
```

## Troubleshooting

**Connection refused?**

- Ensure server is running: `python socket_api.py`
- Check port is available: `netstat -an | grep 5000`

**Slow transcription?**

- Use smaller model: `"tiny"` or `"small"`
- Check system resources (GPU available?)

**Buffer issues?**

- Increase buffer size in `RealTimeAudioProcessor`
- Send audio less frequently

## Hardware Requirements

- **CPU**: 2+ cores recommended
- **RAM**: 2GB minimum (4GB+ recommended)
- **GPU**: Optional but recommended for faster transcription
  - CUDA-capable GPU for faster processing
  - Will automatically use GPU if available

## License

OpenAI Whisper - Apache 2.0
