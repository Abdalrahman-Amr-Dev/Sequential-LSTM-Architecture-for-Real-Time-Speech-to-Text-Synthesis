"""
Original Speech-to-Text Implementation
For real-time API usage, see:
  - websocket_api.py (recommended - web dashboard at http://localhost:5000)
  - socket_api.py (raw socket server)
  - socket_client.py (Python socket client example)
  - advanced_socket_client.py (advanced features like file streaming)
"""

import whisper

# 'base' is fast, 'large' is most accurate
model = whisper.load_model("tiny")

result = model.transcribe("harvard.wav")  

print(result["text"])

# To run the API servers instead:
# >>> python websocket_api.py      # WebSocket + Dashboard
# >>> python socket_api.py          # Raw Socket Server