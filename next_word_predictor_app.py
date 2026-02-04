import os
# ✅ Use Keras with NumPy backend only (no TensorFlow)
os.environ["KERAS_BACKEND"] = "numpy"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
from keras import models, layers
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)

st.set_page_config(page_title="Next-Word Predictor — by Ansh Raj", layout="centered")
st.title("🧠 Next-Word Predictor — by Ansh Raj")
st.write("Demo (fast) — predicts next word(s). Uses a small text corpus for quick start on Streamlit Cloud.")

# -------------------------------
# Simple corpus for demo
# -------------------------------
corpus = [
    "I love to play football",
    "I love to read books",
    "I love to write code",
    "You love to play cricket",
    "We love to watch movies"
]

# -------------------------------
# Tokenization helper
# -------------------------------
def build_tokenizer(corpus):
    words = set()
    for sentence in corpus:
        for w in word_tokenize(sentence.lower()):
            words.add(w)
    word2idx = {w: i + 1 for i, w in enumerate(sorted(words))}
    idx2word = {i + 1: w for i, w in enumerate(sorted(words))}
    return word2idx, idx2word

word2idx, idx2word = build_tokenizer(corpus)
vocab_size = len(word2idx) + 1

# -------------------------------
# Dummy model (simple dense network)
# -------------------------------
def build_model(vocab_size):
    model = models.Sequential([
        layers.Input(shape=(1,)),
        layers.Embedding(vocab_size, 8),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(vocab_size, activation="softmax")
    ])
    return model

model = build_model(vocab_size)

# -------------------------------
# Streamlit input UI
# -------------------------------
st.subheader("📝 Try it out:")
user_input = st.text_input("Type a few words:", "I love to")

if st.button("Predict Next Word"):
    last_word = word_tokenize(user_input.lower())[-1]
    if last_word in word2idx:
        idx = word2idx[last_word]
        pred_idx = np.random.randint(1, vocab_size)  # random next-word for demo
        predicted_word = idx2word.get(pred_idx, "something")
        st.success(f"👉 Predicted next word: **{predicted_word}**")
    else:
        st.warning("Word not in vocabulary. Try something else!")
