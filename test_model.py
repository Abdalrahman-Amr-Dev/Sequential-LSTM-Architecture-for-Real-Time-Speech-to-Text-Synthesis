import numpy as np
import tensorflow as tf
from tensorflow import keras # type: ignore
import librosa

# Load labels (must match training vocabulary)
characters = [c for c in "abcdefghijklmnopqrstuvwxyz' "]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

def transcribe(audio_path):
    model = keras.models.load_model("full_english_asr.h5", custom_objects={'CTCLoss': None})
    
    # 1. Load and convert any format
    audio, _ = librosa.load(audio_path, sr=22050, mono=True) # LJSpeech uses 22050Hz
    
    # 2. Preprocess (Spectrogram)
    stfts = tf.signal.stft(audio, frame_length=256, frame_step=160, fft_length=384)
    spectrogram = tf.abs(stfts)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    
    # Normalize
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    spectrogram = tf.expand_dims(spectrogram, axis=0) # Batch dim

    # 3. Predict & Decode CTC
    prediction = model.predict(spectrogram)
    input_len = np.ones(prediction.shape[0]) * prediction.shape[1]
    # Use greedy decoder to turn probabilities into characters
    results = keras.backend.ctc_decode(prediction, input_length=input_len, greedy=True)[0][0]
    
    # Convert numbers back to text
    output_text = tf.strings.reduce_join(num_to_char(results)).numpy().decode("utf-8")
    print(f"\n--- Transcribed Text ---\n{output_text.strip()}")

# Test with your file
transcribe("my_voice_recording.mp3")