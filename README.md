# Real-Time Speech-to-Text API Documentation

A WebSocket-based server that accepts raw audio streams, maintains a rolling buffer, and performs speech-to-text transcription using OpenAI's Whisper model.

## Overview

This application provides a Flask-SocketIO implementation for real-time speech-to-text processing. The server maintains a 5-second rolling audio buffer and processes transcription requests on-demand.

## Core Components

### Model Initialization

The application loads the Whisper model on startup.

- **Model Selection**: Controlled by the `WHISPER_MODEL` environment variable (defaults to `tiny`)
- **Device**: Runs on CPU by default (standard Whisper behavior unless specified)

### RealTimeAudioProcessor Class

Manages the audio data lifecycle with thread-safe operations on the audio buffer.

#### `__init__(sample_rate=16000, chunk_duration=0.5)`

Initializes the processor with a rolling buffer that holds up to 5 seconds of audio data.

- Sets up the sample rate
- Creates a deque (double-ended queue) to act as a rolling buffer
- Automatically discards oldest data when buffer exceeds capacity

#### `add_audio(audio_bytes)`

Receives raw binary audio data from the WebSocket.

- Converts bytes into 32-bit floating-point NumPy array (Whisper format)
- Appends data to the buffer
- Uses `threading.Lock` for thread-safe concurrent write operations

#### `get_buffered_audio()`

Consolidates the current state of the rolling buffer.

- Converts deque into a single continuous NumPy array
- Returns `None` if buffer is empty

#### `transcribe_buffered()`

Performs speech-to-text conversion on buffered audio.

1. Retrieves current buffer
2. Validates at least 1 second of audio exists (quality assurance)
3. Passes audio to `model.transcribe()`
4. Returns transcribed string or `None` on error/insufficient audio

#### `clear_buffer()`

Resets the audio session by emptying the deque for a new stream.

### SocketIO Event Handlers

Define the API's communication protocol with clients.

#### `@socketio.on('connect')`

- Handles new client connections
- Logs unique Session ID (sid)
- Sends confirmation message to client

#### `@socketio.on('audio_stream')`

- Receives continuous audio chunks from client
- Extracts audio bytes from payload
- Passes data to `processor.add_audio()`
- Emits `buffer_update` with current sample count

#### `@socketio.on('transcribe_request')`

- Triggers server to process current buffer
- Calls `processor.transcribe_buffered()`
- Emits `transcription_result` with text and success boolean

#### `@socketio.on('clear_buffer')`

- Manual buffer management from client side
- Calls `processor.clear_buffer()`
- Notifies client that buffer size is 0

## Data Flow

1. **Connection**: Client connects via WebSocket
2. **Streaming**: Client sends float32 audio chunks via `audio_stream` event
3. **Buffering**: Server maintains last 5 seconds of audio in memory
4. **Processing**: `transcribe_request` triggers Whisper on the 5-second window
5. **Response**: Transcribed text sent back via `transcription_result` event

## Configuration

| Variable        | Description                                                     | Default              |
| --------------- | --------------------------------------------------------------- | -------------------- |
| `WHISPER_MODEL` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) | `tiny`               |
| `SECRET_KEY`    | Flask session security key                                      | `your-secret-key...` |
| `PORT`          | Server listening port                                           | `5000`               |

## Technical Requirements

- **Audio Format**: Raw float32 PCM audio at 16000Hz sample rate
- **Concurrency**: Python threading ensures audio ingestion isn't blocked by transcription computation
