# -*- coding: utf-8 -*-
# TensorFlow NLP module 5
# Run the code cells sequentially after configuring dataset and model paths.

# %% Notebook cell 6
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# %% Notebook cell 7
# Shared constants
VOCAB_SIZE   = 8000
BATCH_SIZE   = 64
MAX_LEN      = 40      # max source/target tokens (excl. [START]/[END])

# Tiny illustration model
TINY_LAYERS  = 1
TINY_D_MODEL = 32
TINY_HEADS   = 2
TINY_DFF     = 64
TINY_SAMPLES = 500
TINY_EPOCHS  = 30

# Full training model
FULL_LAYERS  = 2
FULL_D_MODEL = 128
FULL_HEADS   = 4
FULL_DFF     = 256
FULL_SAMPLES = 10000
FULL_EPOCHS  = 30

# %% Notebook cell 9
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len; self.d_model = d_model
        positions = np.arange(max_len)[:, np.newaxis]
        i_indices = np.arange(d_model // 2)[np.newaxis, :]
        angles    = positions / np.power(10000.0, (2 * i_indices) / d_model)
        pe        = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(angles); pe[:, 1::2] = np.cos(angles)
        self.pe   = tf.cast(pe[np.newaxis, :, :], tf.float32)
    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]
    def get_config(self):
        c = super().get_config(); c.update({"max_len": self.max_len, "d_model": self.d_model}); return c


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model; self.num_heads = num_heads
        self.dff = dff; self.dropout_rate = dropout_rate
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn1 = layers.Dense(dff, activation='relu'); self.ffn2 = layers.Dense(d_model)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6); self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout_rate); self.drop2 = layers.Dropout(dropout_rate)
    def call(self, x, training=False, mask=None):
        a = self.drop1(self.mha(x, x, x, attention_mask=mask), training=training)
        x = self.norm1(x + a)
        f = self.drop2(self.ffn2(self.ffn1(x)), training=training)
        return self.norm2(x + f)
    def get_config(self):
        c = super().get_config(); c.update({"d_model": self.d_model, "num_heads": self.num_heads, "dff": self.dff, "dropout_rate": self.dropout_rate}); return c


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model; self.num_heads = num_heads
        self.dff = dff; self.dropout_rate = dropout_rate
        self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn1 = layers.Dense(dff, activation='relu'); self.ffn2 = layers.Dense(d_model)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout_rate)
        self.drop2 = layers.Dropout(dropout_rate)
        self.drop3 = layers.Dropout(dropout_rate)
    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        a1 = self.drop1(self.mha1(x, x, x, attention_mask=look_ahead_mask), training=training)
        x  = self.norm1(x + a1)
        a2 = self.drop2(self.mha2(x, enc_output, enc_output, attention_mask=padding_mask), training=training)
        x  = self.norm2(x + a2)
        f  = self.drop3(self.ffn2(self.ffn1(x)), training=training)
        return self.norm3(x + f)
    def get_config(self):
        c = super().get_config(); c.update({"d_model": self.d_model, "num_heads": self.num_heads, "dff": self.dff, "dropout_rate": self.dropout_rate}); return c


def create_padding_mask(seq):
    return tf.not_equal(seq, 0)[:, tf.newaxis, :]

def create_look_ahead_mask(size):
    rows = tf.cast(tf.range(size), tf.int32)
    cols = tf.cast(tf.range(size), tf.int32)
    return rows[:, tf.newaxis] >= cols[tf.newaxis, :]

def create_combined_mask(tar):
    seq_len = tf.shape(tar)[1]
    return create_look_ahead_mask(seq_len) & create_padding_mask(tar)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_len, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self._d_model = d_model; self._dropout_rate = dropout_rate
        self._num_layers = num_layers; self._num_heads = num_heads
        self._dff = dff; self._vocab_size = vocab_size; self._max_len = max_len
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate, name=f"enc_{i}") for i in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)
    def call(self, x, training=False, mask=None):
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self._d_model, tf.float32))
        x = self.dropout(self.pos_encoding(x), training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training, mask=mask)
        return x
    def get_config(self):
        c = super().get_config(); c.update({"num_layers": self._num_layers, "d_model": self._d_model, "num_heads": self._num_heads, "dff": self._dff, "vocab_size": self._vocab_size, "max_len": self._max_len, "dropout_rate": self._dropout_rate}); return c


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_len, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self._d_model = d_model; self._dropout_rate = dropout_rate
        self._num_layers = num_layers; self._num_heads = num_heads
        self._dff = dff; self._vocab_size = vocab_size; self._max_len = max_len
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate, name=f"dec_{i}") for i in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)
    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self._d_model, tf.float32))
        x = self.dropout(self.pos_encoding(x), training=training)
        for layer in self.dec_layers:
            x = layer(x, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
        return x
    def get_config(self):
        c = super().get_config(); c.update({"num_layers": self._num_layers, "d_model": self._d_model, "num_heads": self._num_heads, "dff": self._dff, "vocab_size": self._vocab_size, "max_len": self._max_len, "dropout_rate": self._dropout_rate}); return c


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_len, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, max_len, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_len, dropout_rate)
        self.final_layer = layers.Dense(target_vocab_size, name="output_projection")
    def call(self, inputs, training=False):
        inp, tar = inputs
        enc_out = self.encoder(inp, training=training, mask=create_padding_mask(inp))
        dec_out = self.decoder(tar, enc_out, training=training,
                               look_ahead_mask=create_combined_mask(tar),
                               padding_mask=create_padding_mask(inp))
        return self.final_layer(dec_out)


print("All Transformer components defined.")

# %% Notebook cell 11
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

# 4. Shuffle and select samples
# Use N_SAMPLES from the earlier hyperparameter cell.
N_SAMPLES = 500
df_sample = df_merged.sample(
    n=min(N_SAMPLES, len(df_merged)),
    random_state=42
).reset_index(drop=True)

# 5. Extract matching Portuguese and English transcripts
pt_texts = df_sample["transcript_pt"].astype(str).tolist()
en_texts = df_sample["transcript_en"].astype(str).tolist()

print(f"Loaded {len(pt_texts)} matched transcript pairs.")
print(f"Example PT (truncated): {pt_texts[0][:150]}...")
print(f"Example EN (truncated): {en_texts[0][:150]}...")

# %% Notebook cell 13
# Build tokenizers — English gets [START] and [END] markers
en_with_tokens = ['[START] ' + s + ' [END]' for s in en_texts]

pt_vec = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE, output_mode='int', output_sequence_length=MAX_LEN
)
en_vec = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE, output_mode='int', output_sequence_length=MAX_LEN + 2
)
pt_vec.adapt(pt_texts)
en_vec.adapt(en_with_tokens)

pt_tokens = pt_vec(pt_texts).numpy()        # (N, MAX_LEN)
en_tokens = en_vec(en_with_tokens).numpy()  # (N, MAX_LEN + 2)

print(f"PT tokens shape: {pt_tokens.shape}")
print(f"EN tokens shape: {en_tokens.shape}")
print(f"PT vocab size: {len(pt_vec.get_vocabulary())}")
print(f"EN vocab size: {len(en_vec.get_vocabulary())}")

# Locate [START] and [END] IDs for the Translator
en_vocab = en_vec.get_vocabulary()

# Look for the lowercased, unbracketed versions
START_ID = en_vocab.index('start')
END_ID   = en_vocab.index('end')

print(f"[START] token ID: {START_ID}")
print(f"[END] token ID: {END_ID}")

# %% Notebook cell 15
# Training helper — split into teacher-forcing pairs and batch
def make_dataset(pt_toks, en_toks):
    ds = tf.data.Dataset.from_tensor_slices((pt_toks, en_toks))
    return ds.shuffle(len(pt_toks), seed=42).batch(BATCH_SIZE)


# Shared training loop
def run_training(transformer, dataset, epochs, label=''):
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

    losses = []
    for epoch in range(epochs):
        total, n = 0.0, 0
        for inp_b, tar_b in dataset:
            total += float(train_step(inp_b, tar_b))
            n += 1
        avg = total / n
        losses.append(avg)
        print(f"  [{label}] Epoch {epoch + 1}/{epochs}  loss: {avg:.4f}")
    return losses

# %% Notebook cell 17
tiny_ds = make_dataset(pt_tokens[:TINY_SAMPLES], en_tokens[:TINY_SAMPLES])

tiny_transformer = Transformer(
    num_layers=TINY_LAYERS, d_model=TINY_D_MODEL,
    num_heads=TINY_HEADS,   dff=TINY_DFF,
    input_vocab_size=VOCAB_SIZE, target_vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN + 2, name="TinyTransformer"
)

# Build with a dummy forward pass to enable summary
_ = tiny_transformer(
    [tf.zeros((1, MAX_LEN), tf.int32), tf.zeros((1, MAX_LEN + 1), tf.int32)]
)
tiny_transformer.summary()

# %% Notebook cell 19
print("Tiny model training (500 samples, 30 epochs)\n")
tiny_losses = run_training(tiny_transformer, tiny_ds, TINY_EPOCHS, label='Tiny')

plt.figure(figsize=(10, 6))
plt.plot(range(1, TINY_EPOCHS + 1), tiny_losses,
         color='blue', marker='o', label='Training Loss')
plt.title('Tiny Transformer — 500 Pairs, 5 Epochs (Illustration Only)')
plt.xlabel('Epoch')
plt.ylabel('Sparse Categorical Crossentropy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print(f"Final loss: {tiny_losses[-1]:.4f}")

# %% Notebook cell 21
full_ds = make_dataset(pt_tokens, en_tokens)

full_transformer = Transformer(
    num_layers=FULL_LAYERS, d_model=FULL_D_MODEL,
    num_heads=FULL_HEADS,   dff=FULL_DFF,
    input_vocab_size=VOCAB_SIZE, target_vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN + 2, name="FullTransformer"
)

_ = full_transformer(
    [tf.zeros((1, MAX_LEN), tf.int32), tf.zeros((1, MAX_LEN + 1), tf.int32)]
)
full_transformer.summary()

# %% Notebook cell 22
print("Full model training (10k samples, 30 epochs — GPU recommended)\n")
full_losses = run_training(full_transformer, full_ds, FULL_EPOCHS, label='Full')

plt.figure(figsize=(10, 6))
plt.plot(range(1, FULL_EPOCHS + 1), full_losses,
         color='blue', marker='o', label='Training Loss')
plt.title('Full Transformer — 10k Pairs, 10 Epochs')
plt.xlabel('Epoch')
plt.ylabel('Sparse Categorical Crossentropy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print(f"Final loss: {full_losses[-1]:.4f}")

# %% Notebook cell 25
class Translator:
    """
    Autoregressive greedy decoder for a trained Transformer.

    At each step the Decoder attends to the full encoder output and
    all tokens it has generated so far, then picks the single highest-
    probability next token (greedy argmax).
    """

    def __init__(self, transformer, pt_vec, en_vec, max_len=MAX_LEN):
        self.transformer = transformer
        self.pt_vec      = pt_vec
        self.max_len     = max_len

        vocab           = en_vec.get_vocabulary()
        self.start_id   = vocab.index('start')
        self.end_id     = vocab.index('end')
        self.id_to_word = {i: w for i, w in enumerate(vocab)}

    def translate(self, sentence):
        # Step 1: tokenise source sentence → (1, MAX_LEN)
        pt_tokens = self.pt_vec([sentence])

        # Step 2: initialise decoder with [START]
        output = tf.constant([[self.start_id]], dtype=tf.int32)  # (1, 1)

        for _ in range(self.max_len):
            # Step 3: forward pass — logits shape (1, cur_len, vocab_size)
            predictions = self.transformer([pt_tokens, output], training=False)
            next_logits = predictions[:, -1, :]          # last position only
            next_id     = int(tf.argmax(next_logits, axis=-1)[0])  # greedy argmax

            # Step 4: stop at [END]
            if next_id == self.end_id:
                break

            # Step 5: grow the output sequence
            output = tf.concat(
                [output, tf.constant([[next_id]], dtype=tf.int32)], axis=-1
            )

        # Step 6: decode token IDs to words (skip [START])
        token_ids = output.numpy()[0][1:]
        words     = [
            self.id_to_word.get(i, '')
            for i in token_ids
            if self.id_to_word.get(i, '') not in ('', '[UNK]', '[START]', '[END]')
        ]
        return ' '.join(words)


translator = Translator(full_transformer, pt_vec, en_vec)
print("Translator ready.")

# %% Notebook cell 28
test_sentences = [
    "este é o meu primeiro dia.",
    "a tecnologia está mudando o mundo.",
    "obrigado pela sua atenção.",
    "o problema é muito difícil.",
]

print("=" * 60)
for pt_sentence in test_sentences:
    translation = translator.translate(pt_sentence)
    print(f"PT: {pt_sentence}")
    print(f"EN: {translation}")
    print("-" * 60)
