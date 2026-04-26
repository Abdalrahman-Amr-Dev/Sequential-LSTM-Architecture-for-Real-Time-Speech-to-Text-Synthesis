import os
import tarfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Enable Mixed Precision for faster GPU training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# --- DIRECTORY FIX FOR NOTEBOOKS ---
# In notebooks, __file__ doesn't exist. We use os.getcwd() instead.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

cache_dir = os.path.join(script_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)
# -----------------------------------

# Download LJSpeech
data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
data_path = keras.utils.get_file("LJSpeech-1.1.tar.bz2", data_url, cache_dir=cache_dir)

# Path Setup
# The tar file extracts into a folder named "LJSpeech-1.1"
data_dir = os.path.join(os.path.dirname(data_path), "LJSpeech-1.1")
wavs_path = os.path.join(data_dir, "wavs")
metadata_path = os.path.join(data_dir, "metadata.csv")

# Extract via Python only if not already extracted
if not os.path.exists(data_dir):
    print("Extracting files... please wait ~2-3 minutes.")
    with tarfile.open(data_path, mode='r:bz2') as tar:
        tar.extractall(path=os.path.dirname(data_path))
else:
    print("Dataset already extracted.")

# 1. Parse Metadata
# Format: ID | Transcription | Normalized Transcription
with open(metadata_path, encoding="utf-8") as f:
    lines = f.readlines()

split_lines = [line.strip().split("|") for line in lines]
# Ensure we only take lines that have the full 3 parts
split_lines = [x for x in split_lines if len(x) == 3]

filenames = [os.path.join(wavs_path, x[0] + ".wav") for x in split_lines]
transcriptions = [x[2].lower() for x in split_lines]

# 2. Vocabulary Setup
characters = [c for c in "abcdefghijklmnopqrstuvwxyz' "]
char_to_num = layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# 3. Raw Audio Loader
def encode_raw_audio(wav_file, transcription):
    # Read file
    file = tf.io.read_file(wav_file)
    # Decode wav
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # Tokenize transcription
    label = char_to_num(tf.strings.unicode_split(transcription, input_encoding="UTF-8"))
    return audio, label

# 4. Create Dataset
batch_size = 32 
dataset = tf.data.Dataset.from_tensor_slices((filenames, transcriptions))
dataset = (
    dataset.map(encode_raw_audio, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# 5. CTC Loss Function
def CTCLoss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype="int32")
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len,), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len,), dtype="int64")

    loss = tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=label_length,
                          logit_length=input_length, logits_time_major=False)
    return tf.reduce_mean(loss)

# 6. GPU Spectrogram Layer
class SpectrogramLayer(layers.Layer):
    def call(self, inputs):
        # STFT requires float32
        x = tf.cast(inputs, tf.float32)
        stfts = tf.signal.stft(x, frame_length=256, frame_step=160, fft_length=384)
        spectrogram = tf.abs(stfts)
        spectrogram = tf.math.pow(spectrogram, 0.5)

        # Normalization
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        normalized = (spectrogram - means) / (stddevs + 1e-10)

        # Cast back to mixed precision policy (float16)
        return tf.cast(normalized, self.compute_dtype)

# 7. Build DeepSpeech2 Lite
def build_model(output_dim):
    input_audio = layers.Input(shape=(None,), name="input_audio")

    x = SpectrogramLayer()(input_audio)
    x = layers.Reshape((-1, 193, 1))(x)

    # CNN Layers
    x = layers.Conv2D(32, kernel_size=(11, 41), strides=(2, 2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Reshape for RNN (flattening the frequency and channel dimensions)
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # RNN Layers
    for _ in range(2):
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
        x = layers.BatchNormalization()(x)

    # Final Dense (Force float32 for CTC Loss stability)
    output = layers.Dense(output_dim + 1, activation="softmax", dtype='float32')(x)

    return keras.Model(input_audio, output)

model = build_model(output_dim=len(char_to_num.get_vocabulary()))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=CTCLoss)
model.summary()

# 8. Output and Training
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)
checkpoint_path = os.path.join(output_dir, "asr_best_model.h5")

callbacks = [
    keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="loss"),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

# Diagnostic check
if len(filenames) > 0:
    sample_audio_file = filenames[0]
    if not os.path.exists(sample_audio_file):
        print(f"Error: The audio file '{sample_audio_file}' was not found.")
        print("Expected path:", sample_audio_file)
        raise FileNotFoundError("Missing dataset audio file. Check extraction path.")

print("Training starting...")
model.fit(dataset, epochs=50, callbacks=callbacks)

model.save(os.path.join(output_dir, "asr_final_model.h5"))