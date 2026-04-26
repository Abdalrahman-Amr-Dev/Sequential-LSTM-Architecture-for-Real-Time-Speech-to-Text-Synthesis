import os
import tarfile
import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore

# 1. Enable Mixed Precision for faster GPU training
from tensorflow.keras import mixed_precision # type: ignore
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Download LJSpeech
data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "cache")
data_path = keras.utils.get_file("LJSpeech-1.1.tar.bz2", data_url, cache_dir=cache_dir)

# Path Setup
data_dir = os.path.join(script_dir, "datasets", "LJSpeech-1.1")
wavs_path = os.path.join(data_dir, "wavs")
metadata_path = os.path.join(data_dir, "metadata.csv")

# Extract via Python only if not already extracted
if not os.path.exists(data_dir):
    print("Extracting files... please wait ~2-3 minutes.")
    os.makedirs(os.path.dirname(data_dir), exist_ok=True)
    with tarfile.open(data_path, mode='r:bz2') as tar:
        tar.extractall(path=os.path.dirname(data_dir))
else:
    print("Dataset already extracted.")

# 1. Parse Metadata
with open(metadata_path, encoding="utf-8") as f:
    lines = f.readlines()
split_lines = [line.strip().split("|") for line in lines]
filenames = [os.path.join(wavs_path, x[0] + ".wav") for x in split_lines]
transcriptions = [x[2].lower() for x in split_lines]

# 2. Vocabulary Setup
characters = [c for c in "abcdefghijklmnopqrstuvwxyz' "]
char_to_num = layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# 3. Raw Audio Loader
def encode_raw_audio(wav_file, transcription):
    file = tf.io.read_file(wav_file)
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    label = char_to_num(tf.strings.unicode_split(transcription, input_encoding="UTF-8"))
    return audio, label

# 4. Create Dataset
batch_size = 32 # Increased batch size for GPU efficiency
dataset = tf.data.Dataset.from_tensor_slices((filenames, transcriptions))
dataset = (
    dataset.map(encode_raw_audio, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# 1. CTC Loss Function
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

# 2. GPU Spectrogram Layer (Now with float32 casting fix)
class SpectrogramLayer(layers.Layer):
    def call(self, inputs):
        # STFT MUST be float32, so we cast explicitly
        x = tf.cast(inputs, tf.float32)
        stfts = tf.signal.stft(x, frame_length=256, frame_step=160, fft_length=384)
        spectrogram = tf.abs(stfts)
        spectrogram = tf.math.pow(spectrogram, 0.5)

        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        normalized = (spectrogram - means) / (stddevs + 1e-10)

        # Cast back to the global policy (float16) to keep the rest of the model fast
        return tf.cast(normalized, self.compute_dtype)

# 3. Build DeepSpeech2 Lite
def build_model(output_dim):
    input_audio = layers.Input(shape=(None,), name="input_audio")

    x = SpectrogramLayer()(input_audio)
    x = layers.Reshape((-1, 193, 1))(x)

    # CNN Layers
    x = layers.Conv2D(32, kernel_size=(11, 41), strides=(2, 2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Reshape for RNN
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

output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)
checkpoint_path = os.path.join(output_dir, "asr_best_model.h5")

callbacks = [
    keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="loss"),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

print("Training starting... CPU load is now minimal. GPU should be at high utilization.")

# Diagnostic check for the presence of audio files
if len(filenames) > 0:
    sample_audio_file = filenames[0]
    if not os.path.exists(sample_audio_file):
        print(f"Error: The audio file '{sample_audio_file}' was not found.")
        print("Please ensure the dataset (especially the 'wavs' folder) is correctly extracted.")
        print("You might need to delete the local datasets directory and re-run the extraction.")
        raise FileNotFoundError(f"Missing dataset audio file: {sample_audio_file}")

model.fit(dataset, epochs=50, callbacks=callbacks)

model.save(os.path.join(output_dir, "asr_final_model.h5"))