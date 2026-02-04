import os
# ✅ Use Keras with NumPy backend only (no TensorFlow)
os.environ["KERAS_BACKEND"] = "numpy"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
from keras import models, layers, utils
from keras.utils import text_dataset_from_directory
from keras.utils import pad_sequences
import nltk

nltk.download("punkt", quiet=True)

st.set_page_config(page_title="Next-Word Predictor — by Ansh Raj", layout="centered")
st.title("🧠 Next-Word Predictor — by Ansh Raj")
st.write("Demo (fast) — predicts next word(s). Uses a small text corpus for quick start on Streamlit Cloud.")

@st.cache_data
def load_corpus():
    data = """Artificial intelligence is transforming the world.
Machine learning helps computers learn from data.
Deep learning uses neural networks for complex tasks.
Natural language processing allows machines to understand human language.
AI applications are everywhere in our daily lives.
The model in this demo is intentionally small for fast startup and easy sharing."""
    return [s.strip().lower() for s in data.split('\n') if s.strip()]

@st.cache_resource
def build_model(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_seq_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))

    X = input_sequences[:,:-1]
    y = input_sequences[:,-1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_seq_len-1))
    model.add(LSTM(150))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train briefly for demo; small epochs for faster startup
    model.fit(X, y, epochs=80, verbose=0)
    return tokenizer, model, max_seq_len

corpus = load_corpus()
tokenizer, model, max_seq_len = build_model(corpus)

seed = st.text_input('Enter a sentence (seed text):', 'artificial intelligence')
n_words = st.slider('How many words to predict?', 1, 5, 1)

if st.button('Predict'):
    def predict_next_word(seed_text, next_words=1):
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
            preds = model.predict(token_list, verbose=0)[0]
            predicted = int(np.argmax(preds))
            # reverse lookup
            word = None
            for w, i in tokenizer.word_index.items():
                if i == predicted:
                    word = w
                    break
            if word is None:
                break
            seed_text += ' ' + word
        return seed_text

    result = predict_next_word(seed, n_words)
    st.success('**Predicted text:** ' + result)

st.markdown('---')
st.caption('Tip: For production, pretrain on a large corpus and load weights at startup to support many users.')
