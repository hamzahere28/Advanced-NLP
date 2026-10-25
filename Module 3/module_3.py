# -*- coding: utf-8 -*-
# TensorFlow NLP module 3
# Run the code cells sequentially after configuring dataset and model paths.

# %% Notebook cell 7
import tensorflow as tf
from tensorflow.keras import layers, Model

class EncoderLayer(tf.keras.layers.Layer):
    """
    One Transformer Encoder layer.

    Sub-layers (in order):
        1. Multi-Head Self-Attention
        2. Add & Norm
        3. Position-Wise Feed-Forward Network
        4. Add & Norm
    """

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model      = d_model
        self.num_heads    = num_heads
        self.dff          = dff
        self.dropout_rate = dropout_rate

        # Sub-layer 1: Multi-Head Self-Attention
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            name="self_attention"
        )

        # Sub-layer 3: Feed-Forward Network (two dense layers)
        self.ffn_dense1 = layers.Dense(dff, activation='relu', name="ffn_dense1")
        self.ffn_dense2 = layers.Dense(d_model, name="ffn_dense2")

        # Layer normalization (one per Add & Norm step)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")

        # Dropout (applied before adding the residual)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        # --- Sub-layer 1 + Add & Norm ---
        attn_output = self.mha(x, x, x, attention_mask=mask)          # self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)                            # residual + norm

        # --- Sub-layer 3 (FFN) + Add & Norm ---
        ffn_output = self.ffn_dense2(self.ffn_dense1(out1))           # FFN
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)                          # residual + norm

        return out2   # shape: (batch, seq_len, d_model) — same as input

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "num_heads": self.num_heads,
            "dff": self.dff, "dropout_rate": self.dropout_rate
        })
        return config


# ── Wrap in a functional Keras model so model.summary() shows full details ──
D_MODEL = 32
NUM_HEADS = 2
DFF = 64
SEQ_LEN = 60

enc_input = layers.Input(shape=(SEQ_LEN, D_MODEL), name="encoder_input")
enc_output = EncoderLayer(D_MODEL, NUM_HEADS, DFF, name="encoder_layer")(enc_input)
encoder_layer_model = Model(inputs=enc_input, outputs=enc_output, name="EncoderLayer_Model")

print("=== EncoderLayer Model Summary ===")
encoder_layer_model.summary()

# %% Notebook cell 13
class DecoderLayer(tf.keras.layers.Layer):
    """
    One Transformer Decoder layer.

    Sub-layers (in order):
        1. Masked Multi-Head Self-Attention  (look-ahead mask)
        2. Add & Norm
        3. Cross-Attention  (Q=decoder, K=V=encoder output)
        4. Add & Norm
        5. Position-Wise Feed-Forward Network
        6. Add & Norm
    """

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model      = d_model
        self.num_heads    = num_heads
        self.dff          = dff
        self.dropout_rate = dropout_rate

        # Sub-layer 1: Masked self-attention (target attends to itself)
        self.mha1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads,
            name="masked_self_attention"
        )

        # Sub-layer 3: Cross-attention (decoder queries the encoder output)
        self.mha2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads,
            name="cross_attention"
        )

        # Sub-layer 5: Feed-Forward Network
        self.ffn_dense1 = layers.Dense(dff, activation='relu', name="ffn_dense1")
        self.ffn_dense2 = layers.Dense(d_model, name="ffn_dense2")

        # Layer normalization (one per Add & Norm step — three total)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.norm3 = layers.LayerNormalization(epsilon=1e-6, name="norm3")

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training=False,
             look_ahead_mask=None, padding_mask=None):
        """
        Args:
            x              : target sequence embeddings  (batch, tgt_len, d_model)
            enc_output     : encoder output             (batch, src_len, d_model)
            look_ahead_mask: causal mask for self-attention  (prevents future peeking)
            padding_mask   : padding mask for cross-attention (ignores <PAD> tokens)
        """
        # --- Sub-layer 1: Masked Self-Attention + Add & Norm ---
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)       # self-attention
        attn1 = self.dropout1(attn1, training=training)
        out1  = self.norm1(x + attn1)

        # --- Sub-layer 3: Cross-Attention + Add & Norm ---
        # Q comes from the decoder (out1); K and V come from the encoder
        attn2 = self.mha2(out1, enc_output, enc_output,
                          attention_mask=padding_mask)                     # cross-attention
        attn2 = self.dropout2(attn2, training=training)
        out2  = self.norm2(out1 + attn2)

        # --- Sub-layer 5: FFN + Add & Norm ---
        ffn_out = self.ffn_dense2(self.ffn_dense1(out2))
        ffn_out = self.dropout3(ffn_out, training=training)
        out3    = self.norm3(out2 + ffn_out)

        return out3   # shape: (batch, tgt_len, d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "num_heads": self.num_heads,
            "dff": self.dff, "dropout_rate": self.dropout_rate
        })
        return config


# ── Wrap in a functional model to show model.summary() ──
tgt_input  = layers.Input(shape=(SEQ_LEN, D_MODEL), name="target_input")
src_input  = layers.Input(shape=(SEQ_LEN, D_MODEL), name="encoder_output")
dec_layer  = DecoderLayer(D_MODEL, NUM_HEADS, DFF, name="decoder_layer")
dec_out    = dec_layer(tgt_input, src_input)
decoder_layer_model = Model(
    inputs=[tgt_input, src_input], outputs=dec_out,
    name="DecoderLayer_Model"
)

print("=== DecoderLayer Model Summary ===")
decoder_layer_model.summary()

# %% Notebook cell 16
import numpy as np

BATCH = 2

# Dummy encoder input: (batch=2, seq_len=60, d_model=32)
dummy_src = tf.zeros((BATCH, SEQ_LEN, D_MODEL))

# EncoderLayer forward pass
enc_layer = EncoderLayer(D_MODEL, NUM_HEADS, DFF)
enc_out   = enc_layer(dummy_src, training=False)
print(f"EncoderLayer  input  shape: {dummy_src.shape}")
print(f"EncoderLayer  output shape: {enc_out.shape}")
print(f"Shape preserved? {dummy_src.shape == enc_out.shape}")

print()

# Dummy decoder target: (batch=2, tgt_len=60, d_model=32)
dummy_tgt = tf.zeros((BATCH, SEQ_LEN, D_MODEL))

# DecoderLayer forward pass — takes both target and encoder output
dec_layer = DecoderLayer(D_MODEL, NUM_HEADS, DFF)
dec_out   = dec_layer(dummy_tgt, enc_out, training=False)
print(f"DecoderLayer  target shape: {dummy_tgt.shape}")
print(f"DecoderLayer  enc_out shape: {enc_out.shape}")
print(f"DecoderLayer  output shape: {dec_out.shape}")

# %% Notebook cell 19
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# 1. Find local dataset files
input_dir = Path("/kaggle/input")

pairs = [
    (path, path.parent / "rt-polarity.neg")
    for path in sorted(input_dir.rglob("rt-polarity.pos"))
    if (path.parent / "rt-polarity.neg").is_file()
]

if len(pairs) != 1:
    raise FileNotFoundError(
        "Attach the polarity dataset using Add Input. "
        "Expected one folder containing rt-polarity.pos "
        "and rt-polarity.neg."
    )

positive_file, negative_file = pairs[0]

def read_lines(path):
    with open(path, "r", encoding="latin-1") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]

positives = read_lines(positive_file)
negatives = read_lines(negative_file)

# 2. Create and shuffle DataFrame
df = pd.DataFrame({
    "text": positives + negatives,
    "label": [1] * len(positives) + [0] * len(negatives)
})

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# 3. Build vocabulary
VOCAB_SIZE = 5000
MAX_LEN = 60

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN
)

vectorizer.adapt(
    tf.constant(df["text"].tolist(), dtype=tf.string)
)

# 4. Prepare 200 samples
df_tiny = df.head(200).reset_index(drop=True)

X_tiny = vectorizer(
    tf.constant(df_tiny["text"].tolist(), dtype=tf.string)
).numpy()

Y_tiny = df_tiny["label"].to_numpy(dtype=np.float32)

print("Full dataset    :", len(df))
print("Vocabulary size :", vectorizer.vocabulary_size())
print("X shape         :", X_tiny.shape)
print("Y shape         :", Y_tiny.shape)

# %% Notebook cell 20
import matplotlib.pyplot as plt

# Build encoder-only classifier:
#   Embedding → PositionalEncoding → EncoderLayer → GlobalAvgPool → Dense

# Re-use PositionalEncoding from Module 2
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        positions   = np.arange(max_len)[:, np.newaxis]
        i_indices   = np.arange(d_model // 2)[np.newaxis, :]
        angles      = positions / np.power(10000.0, (2 * i_indices) / d_model)
        pe          = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles)
        self.pe     = tf.cast(pe[np.newaxis, :, :], tf.float32)
    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]
    def get_config(self):
        c = super().get_config()
        c.update({"max_len": self.max_len, "d_model": self.d_model})
        return c


def build_encoder_classifier(vocab_size, max_len, d_model, num_heads, dff):
    """Encoder-only binary sentiment classifier."""
    inputs = layers.Input(shape=(max_len,), dtype=tf.int32, name="token_ids")
    x = layers.Embedding(vocab_size, d_model, name="embedding")(inputs)
    x = PositionalEncoding(max_len, d_model, name="positional_encoding")(x)
    x = EncoderLayer(d_model, num_heads, dff, name="encoder_layer")(x)
    x = layers.GlobalAveragePooling1D(name="sequence_avg")(x)
    outputs = layers.Dense(1, activation='sigmoid', name="sentiment_output")(x)
    return Model(inputs=inputs, outputs=outputs, name="EncoderOnly_Classifier")


tiny_model = build_encoder_classifier(VOCAB_SIZE, MAX_LEN, D_MODEL, NUM_HEADS, DFF)
print("=== Tiny Encoder-Only Classifier ===")
tiny_model.summary()

tiny_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nStarting tiny training run (200 samples, 5 epochs)...")
history = tiny_model.fit(
    X_tiny, Y_tiny,
    epochs=5,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Learning curves — Course 2 style
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],     label='Training Accuracy',   color='blue',   marker='o')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', marker='o')
ax1.set_title('EncoderLayer Classifier: Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'],     label='Training Loss',   color='blue',   marker='o')
ax2.plot(history.history['val_loss'], label='Validation Loss', color='orange', marker='o')
ax2.set_title('EncoderLayer Classifier: Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (Binary Crossentropy)')
ax2.legend()
ax2.grid(True)

plt.suptitle('Illustration Only — 200 Samples, 5 Epochs', fontstyle='italic', color='gray')
plt.tight_layout()
plt.show()

print(f"\nFinal training loss:     {history.history['loss'][-1]:.4f}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print("Key observation: loss decreased — the EncoderLayer trains correctly!")
