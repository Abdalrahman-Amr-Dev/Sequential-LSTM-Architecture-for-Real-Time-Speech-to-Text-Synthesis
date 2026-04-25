import os
import numpy as np
import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore

# 1. Dataset Setup (LJSpeech - 24 hours of English)
data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
data_path = keras.utils.get_file("LJSpeech-1.1", data_url, extract=True)
wavs_path = os.path.join(os.path.dirname(data_path), "LJSpeech-1.1/wavs")
metadata_path = os.path.join(os.path.dirname(data_path), "LJSpeech-1.1/metadata.csv")

# Parse metadata
with open(metadata_path, encoding="utf-16" if os.name == 'nt' else "utf-8") as f:
    lines = f.readlines()
split_lines = [line.strip().split("|") for line in lines]
filenames = [os.path.join(wavs_path, x[0] + ".wav") for x in split_lines]
transcriptions = [x[2].lower() for x in split_lines]

# Character mapping (Vocabulary)
characters = [c for c in "abcdefghijklmnopqrstuvwxyz' "]
char_to_num = layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# 2. Preprocessing
def encode_sample(wav_file, transcription):
    file = tf.io.read_file(wav_file)
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # Spectrogram
    stfts = tf.signal.stft(audio, frame_length=256, frame_step=160, fft_length=384)
    spectrogram = tf.abs(stfts)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # Normalization
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    # Label encoding
    label = char_to_num(tf.strings.unicode_split(transcription, input_encoding="UTF-8"))
    return spectrogram, label

# Create Dataset
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices((filenames, transcriptions))
dataset = dataset.map(encode_sample, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 3. Model (DeepSpeech2 Lite)
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return tf.nn.ctc_loss(y_true, y_pred, label_length, input_length, logits_time_major=False)

input_spectrogram = layers.Input(shape=(None, 193), name="input")
x = layers.Reshape((-1, 193, 1))(input_spectrogram)
x = layers.Conv2D(32, kernel_size=(11, 41), strides=(2, 2), padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
output = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation="softmax")(x)

model = keras.Model(input_spectrogram, output)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=CTCLoss)

# 4. Training and Saving
model.fit(dataset.take(500), epochs=1) # Training full English takes time; start small
model.save("full_english_asr.h5")
print("Model saved!")