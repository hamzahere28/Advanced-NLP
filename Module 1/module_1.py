# -*- coding: utf-8 -*-
# TensorFlow NLP module 1
# Run the code cells sequentially after configuring dataset and model paths.

# %% Notebook cell 4
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """Compute scaled dot-product attention."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V
    return output, weights

# Simulate a 3-word sequence with d_k = 4 dimensional Q, K, V vectors
np.random.seed(42)
d_k = 4
words = ["The", "cat", "sleeps"]

Q = np.random.randn(3, d_k)
K = np.random.randn(3, d_k)
V = np.random.randn(3, d_k)

output, attention_weights = scaled_dot_product_attention(Q, K, V)

print("Attention Weight Matrix (each row sums to 1.0):")
print(f"{'':12s} {'Key: The':>12s} {'Key: cat':>12s} {'Key: sleeps':>12s}")
for i, word in enumerate(words):
    row = " ".join([f"{w:12.4f}" for w in attention_weights[i]])
    print(f"Query: {word:<6s} {row}")

print(f"\nRow sums (should all be 1.0): {attention_weights.sum(axis=-1)}")
print(f"Output shape: {output.shape}")

# %% Notebook cell 8
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(attention_weights, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Attention Weight')

ax.set_xticks(range(len(words)))
ax.set_yticks(range(len(words)))
ax.set_xticklabels([f'Key: "{w}"' for w in words])
ax.set_yticklabels([f'Query: "{w}"' for w in words])
ax.set_title("Self-Attention Weight Matrix\n(each row sums to 1.0)")
ax.set_xlabel("Keys (being attended TO)")
ax.set_ylabel("Queries (doing the attending)")

# Annotate each cell with its value
for i in range(len(words)):
    for j in range(len(words)):
        ax.text(j, i, f"{attention_weights[i, j]:.3f}",
                ha='center', va='center', color='black', fontsize=11)

plt.tight_layout()
plt.show()

print("\nWhat does this tell us?")
max_pair = np.unravel_index(attention_weights.argmax(), attention_weights.shape)
print(f"  Highest attention: '{words[max_pair[0]]}' attends most strongly to '{words[max_pair[1]]}' ({attention_weights[max_pair]:.3f})")

# %% Notebook cell 11
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_attention_classifier(seq_len, d_model, num_heads, d_proj=None):
    """
    Binary sentiment classifier using self-attention.
    
    Args:
        seq_len   : maximum sequence length (number of tokens)
        d_model   : input embedding dimension (250 for Wiki-words)
        num_heads : number of parallel attention heads
        d_proj    : if set, projects embeddings to this dimension before attention
    """
    inputs = layers.Input(shape=(seq_len, d_model), name="word_embeddings")
    x = inputs

    # Optional projection: compress large embeddings to a smaller d_model
    if d_proj is not None:
        x = layers.Dense(d_proj, use_bias=False, name="input_projection")(x)
        key_dim = d_proj // num_heads
    else:
        key_dim = d_model // num_heads

    # Self-attention: every word attends to every other word simultaneously
    x = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        name="self_attention"
    )(x, x)

    # Collapse the sequence dimension — identical role to the LSTM's final hidden state
    x = layers.GlobalAveragePooling1D(name="sequence_avg")(x)

    # Binary classification output
    outputs = layers.Dense(1, activation='sigmoid', name="sentiment_output")(x)

    return Model(inputs=inputs, outputs=outputs, name="Attention_Classifier")


# Full-size model — matches the Wiki-words embedding dimension from Course 2
SEQ_LEN  = 59   # max token length from Course 2's Sentiment Dataset
D_MODEL  = 250  # Wiki-words embedding dimension
NUM_HEADS = 2

full_model = build_attention_classifier(SEQ_LEN, D_MODEL, NUM_HEADS)
print("=== Full-Size Attention Classifier ===")
full_model.summary()

# %% Notebook cell 14
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow_hub as hub

# --------------------------------------------------
# 1. Find uploaded positive and negative files
# --------------------------------------------------

input_dir = Path("/kaggle/input")

positive_files = sorted(input_dir.rglob("rt-polarity.pos"))

matching_pairs = [
    (positive, positive.parent / "rt-polarity.neg")
    for positive in positive_files
    if (positive.parent / "rt-polarity.neg").is_file()
]

if len(matching_pairs) != 1:
    raise FileNotFoundError(
        "Expected one folder containing rt-polarity.pos "
        f"and rt-polarity.neg. Found: {matching_pairs}"
    )

positive_file, negative_file = matching_pairs[0]

print("Positive file:", positive_file)
print("Negative file:", negative_file)

# --------------------------------------------------
# 2. Read dataset
# --------------------------------------------------

def read_text_lines(path):
    with open(path, "r", encoding="latin-1") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]

positives = read_text_lines(positive_file)
negatives = read_text_lines(negative_file)

df = pd.DataFrame({
    "text": positives + negatives,
    "label": [1] * len(positives) + [0] * len(negatives),
})

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nTotal samples:", len(df))
print("Positive samples:", len(positives))
print("Negative samples:", len(negatives))

# --------------------------------------------------
# 3. Find attached Wiki-words SavedModel
# --------------------------------------------------

saved_models = sorted(input_dir.rglob("saved_model.pb"))

wiki_models = [
    path.parent
    for path in saved_models
    if "wiki" in str(path).lower()
]

if len(wiki_models) != 1:
    raise FileNotFoundError(
        "Could not uniquely identify the attached Wiki-words model.\n"
        f"SavedModels found: {[str(p.parent) for p in saved_models]}\n"
        "Expand Wiki-words in the Input panel and check its files."
    )

model_path = wiki_models[0]

print("\nLoading local embedding model:", model_path)

embed = hub.load(str(model_path))

# Confirm the model returns 250 features per word
test_embedding = np.asarray(embed(["hello"]), dtype=np.float32)

if test_embedding.shape != (1, 250):
    raise ValueError(
        "Expected Wiki-words 250D embeddings, "
        f"but received shape {test_embedding.shape}."
    )

# --------------------------------------------------
# 4. Prepare 200 samples
# --------------------------------------------------

D_MODEL = 250

# Calculate length using the full dataset
max_len = max(len(text.split()) for text in df["text"])

df_tiny = df.head(200).reset_index(drop=True)

# Preallocate zero-padded input
X_tiny = np.zeros(
    (len(df_tiny), max_len, D_MODEL),
    dtype=np.float32
)

for index, text in enumerate(df_tiny["text"]):
    vectors = np.asarray(
        embed(text.split()),
        dtype=np.float32
    )

    # Place embeddings at the end: left zero-padding
    if len(vectors) > 0:
        X_tiny[index, -len(vectors):, :] = vectors

Y_tiny = df_tiny["label"].to_numpy(dtype=np.float32)

print("\nMax sequence length:", max_len)
print("Tiny X shape:", X_tiny.shape)
print("Tiny Y shape:", Y_tiny.shape)
print("\nPreprocessing completed successfully.")

# %% Notebook cell 15
# # Uses build_attention_classifier from Cell 3
# tiny_model = build_attention_classifier(
#     seq_len=max_len,
#     d_model=250,
#     num_heads=2,
#     d_proj=16
# )

# tiny_model.summary()

# tiny_model.compile(
#     optimizer="adam",
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# tiny_history = tiny_model.fit(
#     X_tiny,
#     Y_tiny,
#     epochs=5,
#     batch_size=16,
#     validation_split=0.2,
#     verbose=1
# )

# %% Notebook cell 17
from tensorflow.keras import layers, Model

def build_attention_classifier(
    seq_len, d_model, num_heads, d_proj=None
):
    inputs = layers.Input(
        shape=(seq_len, d_model),
        name="word_embeddings"
    )

    x = inputs
    attention_dim = d_model

    if d_proj is not None:
        x = layers.Dense(
            d_proj,
            use_bias=False,
            name="input_projection"
        )(x)
        attention_dim = d_proj

    x = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=attention_dim // num_heads,
        name="self_attention"
    )(x, x)

    x = layers.GlobalAveragePooling1D(
        name="sequence_avg"
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="sentiment_output"
    )(x)

    return Model(
        inputs=inputs,
        outputs=outputs,
        name="Attention_Classifier"
    )

print("Model function defined successfully.")

# %% Notebook cell 18
# Build the tiny illustration model
# d_proj=16 compresses the 250-dim embeddings before attention — keeps training fast
tiny_model = build_attention_classifier(
    seq_len=max_len,
    d_model=250,
    num_heads=2,
    d_proj=16      # project 250 → 16 dimensions for a lightweight illustration
)

print("=== Tiny Illustration Model ===")
tiny_model.summary()

tiny_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nStarting tiny training run (200 samples, 5 epochs)...")
tiny_history = tiny_model.fit(
    X_tiny, Y_tiny,
    epochs=5,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# %% Notebook cell 20
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
ax1.plot(tiny_history.history['accuracy'],     label='Training Accuracy',   color='blue',   marker='o')
ax1.plot(tiny_history.history['val_accuracy'], label='Validation Accuracy', color='orange', marker='o')
ax1.set_title('Tiny Attention Model: Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss
ax2.plot(tiny_history.history['loss'],     label='Training Loss',   color='blue',   marker='o')
ax2.plot(tiny_history.history['val_loss'], label='Validation Loss', color='orange', marker='o')
ax2.set_title('Tiny Attention Model: Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (Binary Crossentropy)')
ax2.legend()
ax2.grid(True)

plt.suptitle('Illustration Only — 200 Samples, 5 Epochs', fontstyle='italic', color='gray')
plt.tight_layout()
plt.show()

print("\nKey observation: the training loss decreased — self-attention trains successfully!")
print(f"Final training loss:   {tiny_history.history['loss'][-1]:.4f}")
print(f"Final training accuracy: {tiny_history.history['accuracy'][-1]:.4f}")
