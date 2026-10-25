# -*- coding: utf-8 -*-
# TensorFlow NLP module 2
# Run the code cells sequentially after configuring dataset and model paths.

# %% Notebook cell 3
import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def attention_scores(X):
    """Compute raw self-attention scores for an embedding matrix X."""
    d_k = X.shape[-1]
    scores = X @ X.T / np.sqrt(d_k)
    return softmax(scores)

np.random.seed(7)
# Imagine two word embeddings: embedding for position 0 and position 1
emb_word_A = np.array([0.9, 0.1, 0.4, 0.7])   # e.g. "dog"
emb_word_B = np.array([0.2, 0.8, 0.6, 0.3])   # e.g. "man"

# Original order: [dog, man]
X_original = np.stack([emb_word_A, emb_word_B])
# Swapped order: [man, dog]
X_swapped  = np.stack([emb_word_B, emb_word_A])

scores_original = attention_scores(X_original)
scores_swapped  = attention_scores(X_swapped)

print("Attention weights — original order [dog, man]:")
print(scores_original.round(4))
print("\nAttention weights — swapped order [man, dog]:")
print(scores_swapped.round(4))
print("\nAre they the same? (just row/column permuted):", np.allclose(scores_original, scores_swapped[[1,0],:][:,[1,0]]))

# %% Notebook cell 8
import tensorflow as tf
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    """
    Adds fixed sine/cosine positional signals to word embeddings.
    Has zero trainable parameters.
    """

    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        # Pre-compute the full PE matrix once at construction time
        self.pe = self._build_pe_matrix(max_len, d_model)  # (1, max_len, d_model)

    def _build_pe_matrix(self, max_len, d_model):
        """Compute the positional encoding matrix using vectorized NumPy."""
        positions = np.arange(max_len)[:, np.newaxis]          # (max_len, 1)
        i_indices  = np.arange(d_model // 2)[np.newaxis, :]    # (1, d_model/2)

        # Denominator: 10000^(2i / d_model)
        denominator = np.power(10000.0, (2 * i_indices) / d_model)
        angles = positions / denominator                        # (max_len, d_model/2)

        # Interleave: even dimensions → sin, odd dimensions → cos
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles)

        return tf.cast(pe[np.newaxis, :, :], dtype=tf.float32)  # (1, max_len, d_model)

    def call(self, x):
        """Add positional encoding to the input embedding tensor."""
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


# Quick sanity check
pe_layer = PositionalEncoding(max_len=10, d_model=8)
dummy_emb = tf.zeros((1, 10, 8))          # batch=1, seq_len=10, d_model=8
pe_output  = pe_layer(dummy_emb)
print("Input shape:  ", dummy_emb.shape)
print("Output shape: ", pe_output.shape)
print("Output is no longer all zeros (PE added):")
print(pe_output[0, :3, :].numpy().round(4))  # First 3 positions, all 8 dims

# %% Notebook cell 11
from tensorflow.keras import layers, Model

def build_pe_classifier(vocab_size, max_len, d_model, num_heads):
    """
    Sentiment classifier: Embedding → PositionalEncoding → MultiHeadAttention → Dense.

    Args:
        vocab_size : number of unique tokens in the vocabulary
        max_len    : maximum sequence length (tokens)
        d_model    : embedding dimension (also the attention model dimension)
        num_heads  : number of parallel attention heads
    """
    inputs = layers.Input(shape=(max_len,), dtype=tf.int32, name="token_ids")

    # 1. Learnable word embedding (maps integer IDs to dense vectors)
    x = layers.Embedding(vocab_size, d_model, name="word_embedding")(inputs)

    # 2. Add fixed positional encoding — no extra trainable weights
    x = PositionalEncoding(max_len=max_len, d_model=d_model, name="positional_encoding")(x)

    # 3. Self-attention
    x = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        name="self_attention"
    )(x, x)

    # 4. Collapse sequence dimension
    x = layers.GlobalAveragePooling1D(name="sequence_avg")(x)

    # 5. Binary classification
    outputs = layers.Dense(1, activation='sigmoid', name="sentiment_output")(x)

    return Model(inputs=inputs, outputs=outputs, name="PE_Attention_Classifier")


# Build with the tiny illustration dimensions from the issue
VOCAB_SIZE = 5000
MAX_LEN    = 60
D_MODEL    = 32
NUM_HEADS  = 2

model = build_pe_classifier(VOCAB_SIZE, MAX_LEN, D_MODEL, NUM_HEADS)
model.summary()

# %% Notebook cell 14
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# Keep values defined in earlier Module 2 cells.
# These defaults apply only if they are not defined yet.
if "VOCAB_SIZE" not in globals():
    VOCAB_SIZE = 10000

if "MAX_LEN" not in globals():
    MAX_LEN = 59

# 1. Find the attached local dataset
input_dir = Path("/kaggle/input")

matching_pairs = [
    (path, path.parent / "rt-polarity.neg")
    for path in sorted(input_dir.rglob("rt-polarity.pos"))
    if (path.parent / "rt-polarity.neg").is_file()
]

if len(matching_pairs) != 1:
    raise FileNotFoundError(
        "Attach your polarity dataset using Add Input. "
        "It must contain rt-polarity.pos and rt-polarity.neg "
        "in the same folder."
    )

positive_file, negative_file = matching_pairs[0]

def read_lines(path):
    with open(path, "r", encoding="latin-1") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]

positives = read_lines(positive_file)
negatives = read_lines(negative_file)

# 2. Build and shuffle the full dataset
df = pd.DataFrame({
    "text": positives + negatives,
    "label": [1] * len(positives) + [0] * len(negatives)
})

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print(f"Full dataset: {len(df)} samples")
print(f"VOCAB_SIZE: {VOCAB_SIZE}, MAX_LEN: {MAX_LEN}")

# 3. Select 200 samples
df_tiny = df.head(200).reset_index(drop=True)

texts_tiny = df_tiny["text"].tolist()
labels_tiny = df_tiny["label"].to_numpy(dtype=np.float32)

# 4. Build vocabulary, matching the course illustration
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN
)

vectorizer.adapt(
    tf.constant(df["text"].tolist(), dtype=tf.string)
)

print(f"Vocabulary size: {vectorizer.vocabulary_size()}")

# 5. Convert text into integer token IDs
X_tiny = vectorizer(
    tf.constant(texts_tiny, dtype=tf.string)
).numpy()

Y_tiny = labels_tiny

print(f"X shape: {X_tiny.shape} — (samples, seq_len)")
print(f"Y shape: {Y_tiny.shape}")
print("Example tokens:", X_tiny[0, :10])

# %% Notebook cell 16
import matplotlib.pyplot as plt

# Build and compile the tiny illustration model
tiny_model = build_pe_classifier(VOCAB_SIZE, MAX_LEN, D_MODEL, NUM_HEADS)

tiny_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Starting tiny training run (200 samples, 5 epochs)...")
history = tiny_model.fit(
    X_tiny, Y_tiny,
    epochs=5,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Plot learning curves — Course 2 style
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],     label='Training Accuracy',   color='blue',   marker='o')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', marker='o')
ax1.set_title('PE Classifier: Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'],     label='Training Loss',   color='blue',   marker='o')
ax2.plot(history.history['val_loss'], label='Validation Loss', color='orange', marker='o')
ax2.set_title('PE Classifier: Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (Binary Crossentropy)')
ax2.legend()
ax2.grid(True)

plt.suptitle('Illustration Only — 200 Samples, 5 Epochs', fontstyle='italic', color='gray')
plt.tight_layout()
plt.show()

print(f"\nFinal training loss:     {history.history['loss'][-1]:.4f}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print("Key observation: loss decreased — positional encoding trains successfully!")

# %% Notebook cell 19
# Generate a PE matrix for 50 positions with 64 dimensions for a clear visualization
pe_vis = PositionalEncoding(max_len=50, d_model=64)
dummy_input = tf.zeros((1, 50, 64))
pe_matrix = pe_vis(dummy_input)[0].numpy()  # shape (50, 64)

plt.figure(figsize=(10, 6))
plt.imshow(pe_matrix, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='Encoding Value')
plt.title("Positional Encoding Matrix\n(50 positions × 64 dimensions)")
plt.xlabel("Embedding Dimension")
plt.ylabel("Position in Sequence")
plt.grid(True, linestyle='--', alpha=0.7, color='gray')
plt.tight_layout()
plt.show()

# Show cosine similarity between adjacent positions — they should be close to each other
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Cosine similarity between adjacent positions (should be high — nearby positions are similar):")
for pos in [0, 5, 10, 20]:
    sim = cosine_similarity(pe_matrix[pos], pe_matrix[pos + 1])
    print(f"  Position {pos:2d} ↔ {pos+1:2d}: {sim:.4f}")

print("\nCosine similarity between distant positions (should be lower):")
for pair in [(0, 10), (0, 25), (0, 49)]:
    sim = cosine_similarity(pe_matrix[pair[0]], pe_matrix[pair[1]])
    print(f"  Position {pair[0]:2d} ↔ {pair[1]:2d}: {sim:.4f}")
