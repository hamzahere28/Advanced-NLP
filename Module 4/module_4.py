# -*- coding: utf-8 -*-
# TensorFlow NLP module 4
# Run the code cells sequentially after configuring dataset and model paths.

# %% Notebook cell 4
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

print(f"TensorFlow version: {tf.__version__}")

# %% Notebook cell 5
# Tiny hyperparameters — deliberately small for illustration
NUM_LAYERS   = 1
D_MODEL      = 32
NUM_HEADS    = 2
DFF          = 64
DROPOUT_RATE = 0.1
MAX_LEN      = 40     # max source/target tokens
VOCAB_SIZE   = 4000
N_SAMPLES    = 500
BATCH_SIZE   = 16
EPOCHS       = 5

# %% Notebook cell 7
class PositionalEncoding(tf.keras.layers.Layer):
    """Fixed sine/cosine positional encoding — zero trainable parameters."""

    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        positions    = np.arange(max_len)[:, np.newaxis]
        i_indices    = np.arange(d_model // 2)[np.newaxis, :]
        angles       = positions / np.power(10000.0, (2 * i_indices) / d_model)
        pe           = np.zeros((max_len, d_model))
        pe[:, 0::2]  = np.sin(angles)
        pe[:, 1::2]  = np.cos(angles)
        self.pe      = tf.cast(pe[np.newaxis, :, :], tf.float32)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        c = super().get_config()
        c.update({"max_len": self.max_len, "d_model": self.d_model})
        return c


class EncoderLayer(tf.keras.layers.Layer):
    """MHA → Add&Norm → FFN → Add&Norm."""

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model      = d_model
        self.num_heads    = num_heads
        self.dff          = dff
        self.dropout_rate = dropout_rate
        self.mha          = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, name="self_attention"
        )
        self.ffn_dense1   = layers.Dense(dff, activation='relu', name="ffn_dense1")
        self.ffn_dense2   = layers.Dense(d_model, name="ffn_dense2")
        self.norm1        = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2        = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.dropout1     = layers.Dropout(dropout_rate)
        self.dropout2     = layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        attn_out = self.mha(x, x, x, attention_mask=mask)
        attn_out = self.dropout1(attn_out, training=training)
        out1     = self.norm1(x + attn_out)
        ffn_out  = self.ffn_dense2(self.ffn_dense1(out1))
        ffn_out  = self.dropout2(ffn_out, training=training)
        return self.norm2(out1 + ffn_out)

    def get_config(self):
        c = super().get_config()
        c.update({"d_model": self.d_model, "num_heads": self.num_heads,
                  "dff": self.dff, "dropout_rate": self.dropout_rate})
        return c


class DecoderLayer(tf.keras.layers.Layer):
    """Masked-MHA → Add&Norm → Cross-MHA → Add&Norm → FFN → Add&Norm."""

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model      = d_model
        self.num_heads    = num_heads
        self.dff          = dff
        self.dropout_rate = dropout_rate
        self.mha1         = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, name="masked_self_attention"
        )
        self.mha2         = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, name="cross_attention"
        )
        self.ffn_dense1   = layers.Dense(dff, activation='relu', name="ffn_dense1")
        self.ffn_dense2   = layers.Dense(d_model, name="ffn_dense2")
        self.norm1        = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2        = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.norm3        = layers.LayerNormalization(epsilon=1e-6, name="norm3")
        self.dropout1     = layers.Dropout(dropout_rate)
        self.dropout2     = layers.Dropout(dropout_rate)
        self.dropout3     = layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training=False,
             look_ahead_mask=None, padding_mask=None):
        attn1   = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        attn1   = self.dropout1(attn1, training=training)
        out1    = self.norm1(x + attn1)
        attn2   = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        attn2   = self.dropout2(attn2, training=training)
        out2    = self.norm2(out1 + attn2)
        ffn_out = self.ffn_dense2(self.ffn_dense1(out2))
        ffn_out = self.dropout3(ffn_out, training=training)
        return self.norm3(out2 + ffn_out)

    def get_config(self):
        c = super().get_config()
        c.update({"d_model": self.d_model, "num_heads": self.num_heads,
                  "dff": self.dff, "dropout_rate": self.dropout_rate})
        return c


print("PositionalEncoding, EncoderLayer, DecoderLayer defined.")

# %% Notebook cell 9
def create_padding_mask(seq):
    """
    Boolean: True = real token, False = padding (token ID 0).
    Outpu
    t shape: (batch, 1, seq_len) — broadcasts over query positions.
    """
    return tf.not_equal(seq, 0)[:, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Boolean lower-triangular matrix.
    True = position is reachable (past/current), False = future (blocked).
    Output shape: (size, size).
    """
    rows = tf.cast(tf.range(size), tf.int32)
    cols = tf.cast(tf.range(size), tf.int32)
    return rows[:, tf.newaxis] >= cols[tf.newaxis, :]


def create_combined_mask(tar):
    """
    Combines look-ahead causality with padding for decoder self-attention.
    Output shape: (batch, tgt_len, tgt_len).
    """
    seq_len    = tf.shape(tar)[1]
    look_ahead = create_look_ahead_mask(seq_len)  # (tgt_len, tgt_len)
    pad_mask   = create_padding_mask(tar)          # (batch, 1, tgt_len)
    return look_ahead & pad_mask                   # (batch, tgt_len, tgt_len)

# %% Notebook cell 11
# Visualize all three masks with a sample sequence [7, 3, 5, PAD, PAD]
sample_seq = tf.constant([[7, 3, 5, 0, 0]])
pad_mask   = create_padding_mask(sample_seq)    # (1, 1, 5)
look_ahead = create_look_ahead_mask(5)          # (5, 5)
combined   = create_combined_mask(sample_seq)   # (1, 5, 5)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
token_labels = ['7', '3', '5', 'PAD', 'PAD']

axes[0].imshow(tf.cast(pad_mask[0], tf.float32).numpy(), cmap='Blues', vmin=0, vmax=1)
axes[0].set_title('Padding Mask\n(1=attend, 0=ignore padding)')
axes[0].set_xlabel('Key Position')
axes[0].set_yticks([0]); axes[0].set_yticklabels(['query (broadcast)'])
axes[0].set_xticks(range(5)); axes[0].set_xticklabels(token_labels)
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].imshow(tf.cast(look_ahead, tf.float32).numpy(), cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Look-Ahead Mask\n(1=attend, 0=future blocked)')
axes[1].set_xlabel('Key Position'); axes[1].set_ylabel('Query Position')
axes[1].grid(True, linestyle='--', alpha=0.5)

axes[2].imshow(tf.cast(combined[0], tf.float32).numpy(), cmap='Blues', vmin=0, vmax=1)
axes[2].set_title('Combined Mask\n(causal + padding)')
axes[2].set_xlabel('Key Position'); axes[2].set_ylabel('Query Position')
axes[2].set_xticks(range(5)); axes[2].set_xticklabels(token_labels)
axes[2].grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Transformer Masking — Three Masks at a Glance', fontsize=13)
plt.tight_layout()
plt.show()

print(f"Padding mask shape:    {pad_mask.shape}")
print(f"Look-ahead mask shape: {look_ahead.shape}")
print(f"Combined mask shape:   {combined.shape}")

# %% Notebook cell 14
class Encoder(tf.keras.layers.Layer):
    """
    Full Encoder stack:
    Embedding → √d scaling → PositionalEncoding → N × EncoderLayer.
    """

    def __init__(self, num_layers, d_model, num_heads, dff,
                 vocab_size, max_len, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self._num_layers   = num_layers
        self._d_model      = d_model
        self._num_heads    = num_heads
        self._dff          = dff
        self._vocab_size   = vocab_size
        self._max_len      = max_len
        self._dropout_rate = dropout_rate
        self.embedding     = layers.Embedding(vocab_size, d_model)
        self.pos_encoding  = PositionalEncoding(max_len, d_model)
        self.enc_layers    = [
            EncoderLayer(d_model, num_heads, dff, dropout_rate, name=f"enc_layer_{i}")
            for i in range(num_layers)
        ]
        self.dropout       = layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        x  = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self._d_model, tf.float32))
        x  = self.pos_encoding(x)
        x  = self.dropout(x, training=training)
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training, mask=mask)
        return x

    def get_config(self):
        c = super().get_config()
        c.update({
            "num_layers": self._num_layers, "d_model": self._d_model,
            "num_heads":  self._num_heads,  "dff": self._dff,
            "vocab_size": self._vocab_size, "max_len": self._max_len,
            "dropout_rate": self._dropout_rate,
        })
        return c


# Quick shape check
enc_test   = Encoder(num_layers=1, d_model=D_MODEL, num_heads=NUM_HEADS,
                     dff=DFF, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
dummy_src  = tf.zeros((2, MAX_LEN), dtype=tf.int32)
enc_out    = enc_test(dummy_src)
print(f"Encoder input shape:  {dummy_src.shape}")
print(f"Encoder output shape: {enc_out.shape}")

# %% Notebook cell 17
class Decoder(tf.keras.layers.Layer):
    """
    Full Decoder stack:
    Embedding → √d scaling → PositionalEncoding → N × DecoderLayer.
    """

    def __init__(self, num_layers, d_model, num_heads, dff,
                 vocab_size, max_len, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self._num_layers   = num_layers
        self._d_model      = d_model
        self._num_heads    = num_heads
        self._dff          = dff
        self._vocab_size   = vocab_size
        self._max_len      = max_len
        self._dropout_rate = dropout_rate
        self.embedding     = layers.Embedding(vocab_size, d_model)
        self.pos_encoding  = PositionalEncoding(max_len, d_model)
        self.dec_layers    = [
            DecoderLayer(d_model, num_heads, dff, dropout_rate, name=f"dec_layer_{i}")
            for i in range(num_layers)
        ]
        self.dropout       = layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training=False,
             look_ahead_mask=None, padding_mask=None):
        x  = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self._d_model, tf.float32))
        x  = self.pos_encoding(x)
        x  = self.dropout(x, training=training)
        for dec_layer in self.dec_layers:
            x = dec_layer(x, enc_output, training=training,
                          look_ahead_mask=look_ahead_mask,
                          padding_mask=padding_mask)
        return x

    def get_config(self):
        c = super().get_config()
        c.update({
            "num_layers": self._num_layers, "d_model": self._d_model,
            "num_heads":  self._num_heads,  "dff": self._dff,
            "vocab_size": self._vocab_size, "max_len": self._max_len,
            "dropout_rate": self._dropout_rate,
        })
        return c


# Quick shape check
dec_test  = Decoder(num_layers=1, d_model=D_MODEL, num_heads=NUM_HEADS,
                    dff=DFF, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
dummy_tgt = tf.zeros((2, MAX_LEN), dtype=tf.int32)
dec_out   = dec_test(dummy_tgt, enc_out)
print(f"Decoder target shape:    {dummy_tgt.shape}")
print(f"Decoder enc_output shape:{enc_out.shape}")
print(f"Decoder output shape:    {dec_out.shape}")

# %% Notebook cell 20
class Transformer(tf.keras.Model):
    """
    Complete Transformer: Encoder + Decoder + output Dense projection.

    Inputs:
        inp — source token IDs, shape (batch, src_len)
        tar — target token IDs shifted right (teacher forcing), shape (batch, tgt_len)
    Output:
        logits of shape (batch, tgt_len, target_vocab_size)
    """

    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, max_len,
                 dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder     = Encoder(num_layers, d_model, num_heads, dff,
                                   input_vocab_size, max_len, dropout_rate)
        self.decoder     = Decoder(num_layers, d_model, num_heads, dff,
                                   target_vocab_size, max_len, dropout_rate)
        self.final_layer = layers.Dense(target_vocab_size, name="output_projection")

    def call(self, inputs, training=False):
        inp, tar = inputs

        enc_mask       = create_padding_mask(inp)   # (batch, 1, src_len)
        dec_self_mask  = create_combined_mask(tar)  # (batch, tgt_len, tgt_len)
        dec_cross_mask = create_padding_mask(inp)   # (batch, 1, src_len)

        enc_output = self.encoder(inp, training=training, mask=enc_mask)
        dec_output = self.decoder(tar, enc_output, training=training,
                                  look_ahead_mask=dec_self_mask,
                                  padding_mask=dec_cross_mask)
        return self.final_layer(dec_output)         # (batch, tgt_len, target_vocab_size)

# %% Notebook cell 23
transformer = Transformer(
    num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=NUM_HEADS, dff=DFF,
    input_vocab_size=VOCAB_SIZE, target_vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN + 2, name="Transformer"
)

# Dummy forward pass to build the model
dummy_inp = tf.zeros((2, MAX_LEN),     dtype=tf.int32)
dummy_tar = tf.zeros((2, MAX_LEN + 1), dtype=tf.int32)
out       = transformer([dummy_inp, dummy_tar], training=False)

transformer.summary()

print(f"\nEncoder input shape:  {dummy_inp.shape}")
print(f"Decoder input shape:  {dummy_tar.shape}")
print(f"Output logits shape:  {out.shape}")

# %% Notebook cell 26
import pandas as pd
import os

print("Loading TED Ultimate Dataset from Kaggle Inputs...")

# 1. Use the exact file paths you found in the Kaggle Input pane
pt_file_path = '/kaggle/input/datasets/miguelcorraljr/ted-ultimate-dataset/2020-05-01/ted_talks_pt-br.csv'
en_file_path = '/kaggle/input/datasets/miguelcorraljr/ted-ultimate-dataset/2020-05-01/ted_talks_en.csv'

# Note: If the above paths ever throw an error, standard Kaggle inputs sometimes drop the author name. 
# In that case, you would just use: '/kaggle/input/ted-ultimate-dataset/2020-05-01/ted_talks_pt-br.csv'

# 2. Read the CSV files
df_pt = pd.read_csv(pt_file_path)
df_en = pd.read_csv(en_file_path)

# 3. Merge on 'talk_id' to align the translations perfectly
df_merged = pd.merge(df_pt[['talk_id', 'transcript']], 
                     df_en[['talk_id', 'transcript']], 
                     on='talk_id', 
                     suffixes=('_pt', '_en'))

# Drop any rows missing a transcript just to be safe
df_merged = df_merged.dropna(subset=['transcript_pt', 'transcript_en'])

# 4. Extract the samples
# We shuffle with a random state to match your example's behavior, 
# then take N_SAMPLES to keep the training loop fast
df_sample = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
pt_texts = df_sample['transcript_pt'].head(N_SAMPLES).astype(str).tolist()
en_texts = df_sample['transcript_en'].head(N_SAMPLES).astype(str).tolist()

print(f"Loaded {len(pt_texts)} aligned pairs.")
print(f"Example PT (truncated): {pt_texts[0][:150]}...")
print(f"Example EN (truncated): {en_texts[0][:150]}...")


# %% Notebook cell 28
# Tokenize using TextVectorization
# English gets [START] and [END] markers for teacher forcing
en_with_tokens = ['[START] ' + s + ' [END]' for s in en_texts]

pt_vec = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE, output_mode='int', output_sequence_length=MAX_LEN
)
en_vec = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE, output_mode='int', output_sequence_length=MAX_LEN + 2
)

pt_vec.adapt(pt_texts)
en_vec.adapt(en_with_tokens)

pt_tokens = pt_vec(pt_texts).numpy()          # (N, MAX_LEN)
en_tokens = en_vec(en_with_tokens).numpy()    # (N, MAX_LEN + 2)

print(f"PT tokens shape: {pt_tokens.shape}")
print(f"EN tokens shape: {en_tokens.shape}")
print(f"\nPT vocabulary size: {len(pt_vec.get_vocabulary())}")
print(f"EN vocabulary size: {len(en_vec.get_vocabulary())}")

# %% Notebook cell 30
# Build tf.data.Dataset — batched and shuffled
dataset = tf.data.Dataset.from_tensor_slices((pt_tokens, en_tokens))
dataset = dataset.shuffle(N_SAMPLES, seed=42).batch(BATCH_SIZE)

print(f"Number of batches: {len(list(dataset))}")
print(f"Batch shapes (inp, tar): {next(iter(dataset))[0].shape}, {next(iter(dataset))[1].shape}")

# %% Notebook cell 31
# Custom training loop with teacher forcing
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(inp, tar):
    tar_inp  = tar[:, :-1]   # decoder input  — drop last token
    tar_real = tar[:, 1:]    # target labels  — drop [START]
    with tf.GradientTape() as tape:
        predictions = transformer([inp, tar_inp], training=True)
        loss        = loss_fn(tar_real, predictions)
    grads = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(grads, transformer.trainable_variables))
    return loss


print("Training Transformer — 500 pairs, 5 epochs (illustration only)\n")
train_losses = []
for epoch in range(EPOCHS):
    epoch_loss, n_batches = 0.0, 0
    for inp_batch, tar_batch in dataset:
        batch_loss  = train_step(inp_batch, tar_batch)
        epoch_loss += float(batch_loss)
        n_batches  += 1
    avg = epoch_loss / n_batches
    train_losses.append(avg)
    print(f"Epoch {epoch + 1}/{EPOCHS}  loss: {avg:.4f}")

# %% Notebook cell 33
# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_losses, color='blue', marker='o', label='Training Loss')
plt.title('Transformer Training Loss — 500 Pairs, 5 Epochs (Illustration Only)')
plt.xlabel('Epoch')
plt.ylabel('Sparse Categorical Crossentropy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nInitial loss: {train_losses[0]:.4f}")
print(f"Final loss:   {train_losses[-1]:.4f}")
print("Decreasing loss confirms the full Transformer pipeline is wired correctly.")
