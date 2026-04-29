# Quick Start Guide

## 🚀 Get Started in 2 Minutes

### Prerequisites

```bash
pip install -r requirements.txt
```

## Option 1: WebSocket API (Easy - Recommended)

### Terminal 1: Start Server

```bash
python websocket_api.py
```

### Open Dashboard

Visit: **http://localhost:5000**

Click "Start Recording" and see real-time transcription!

---

## Option 2: Raw Socket API (Advanced)

### Terminal 1: Start Server

```bash
python socket_api.py
```

### Terminal 2: Test Client

```bash
python socket_client.py
```

---

## Option 3: Advanced Socket Client (File Streaming)

### Terminal 1: Start Server

```bash
python socket_api.py
```

### Terminal 2: Stream a WAV File

```python
from advanced_socket_client import AdvancedSocketClient

client = AdvancedSocketClient()
client.connect()
client.stream_wav_file("your_audio.wav")
result = client.transcribe()
print(result['text'])
client.close()
```

---

## 🔧 Customization

### Change API Port

Edit `socket_api.py` or `websocket_api.py`:

```python
SocketServer(host='0.0.0.0', port=8000)  # Change 5000 to 8000
```

### Use Faster Model

Edit the model loading line:

```python
model = whisper.load_model("tiny")   # Fastest
model = whisper.load_model("small")  # Fast
model = whisper.load_model("base")   # Balanced (default)
model = whisper.load_model("medium") # More accurate
model = whisper.load_model("large")  # Most accurate
```

### Change Buffer Duration

Edit `RealTimeAudioProcessor`:

```python
RealTimeAudioProcessor(chunk_duration=1.0)  # 1 second chunks
```

---

## 🐛 Troubleshooting

| Problem                                        | Solution                                                                          |
| ---------------------------------------------- | --------------------------------------------------------------------------------- |
| `Address already in use`                       | Port 5000 is taken. Change port or kill process: `lsof -ti:5000 \| xargs kill -9` |
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt`                                             |
| `Connection refused`                           | Server not running. Run `python socket_api.py` or `python websocket_api.py`       |
| `Slow transcription`                           | Use smaller model (`tiny` or `small`)                                             |

---

## 📊 Architecture Overview

```
┌─────────────────────┐
│   Client/Browser    │
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │   Socket/   │
    │  WebSocket  │
    └──────┬──────┘
           │
    ┌──────▼──────────────┐
    │  Audio Buffer      │
    │  (5 seconds)       │
    └──────┬──────────────┘
           │
    ┌──────▼──────────────┐
    │ Whisper Model      │
    │ Transcription      │
    └────────────────────┘
```

---

## 💡 Usage Examples

### Simple WebSocket (Browser)

```javascript
const socket = io("http://localhost:5000");
socket.emit("audio_stream", { audio: audioBuffer });
socket.on("transcription_result", (data) => console.log(data.text));
```

### Python Socket Client

```python
from socket_client import SocketClient
client = SocketClient()
client.connect()
client.send_audio(audio_data)
result = client.transcribe()
print(result['text'])
```

### Stream WAV File

```python
from advanced_socket_client import AdvancedSocketClient
client = AdvancedSocketClient()
client.connect()
client.stream_wav_file("audio.wav")
result = client.transcribe()
print(result['text'])
client.close()
```

---

## 📚 API Reference

### WebSocket Events

- `audio_stream` - Send audio
- `transcribe_request` - Transcribe
- `clear_buffer` - Clear buffer
- `transcription_result` - Get result
- `buffer_update` - Buffer status

### Socket Commands

- `{"command": "audio", "data": "base64_audio"}`
- `{"command": "transcribe"}`
- `{"command": "clear"}`
- `{"command": "status"}`

---

## 🎯 Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Start server: `python websocket_api.py`
3. ✅ Open dashboard: http://localhost:5000
4. ✅ Test recording and transcription
5. ✅ Integrate into your application

---

For detailed documentation, see [README.md](README.md)
