import os
import re
import logging
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, request, render_template
import joblib
import feedparser
from urllib.parse import quote_plus
from difflib import SequenceMatcher

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Download stopwords if not present
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# Global model/tokenizer
model = None
tokenizer = None

# ---------- TEXT CLEANING ----------
def clean_text(text):
    txt = text.lower()
    txt = re.sub(r"[^\w\s]", "", txt)
    return ' '.join(w for w in txt.split() if w not in STOPWORDS)

# ---------- DATA LOADING ----------
def load_data():
    cols = ['id','label','statement','subject','speaker','job_title','state','party',
            'barely_true','false','half_true','mostly_true','pants_on_fire','context']
    df = pd.read_csv('train.tsv', sep='\t', names=cols)
    mapping = {'true':1,'mostly-true':1,'half-true':1,'barely-true':0,'false':0,'pants-fire':0}
    df['binary'] = df['label'].map(mapping)
    df = df.dropna(subset=['statement','binary'])
    df['cleaned'] = df['statement'].apply(clean_text)
    return df

# ---------- TOKENIZER & GLOVE ----------
def prepare_tokenizer(texts):
    t = Tokenizer(num_words=10000, oov_token='<OOV>')
    t.fit_on_texts(texts)
    joblib.dump(t, 'tokenizer.joblib')
    return t, len(t.word_index) + 1

def load_glove_embeddings(glove_path, tokenizer, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i < vocab_size:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix, vocab_size

# ---------- MODEL ----------
def build_model(vocab_size, embedding_matrix):
    m = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix],
                  input_length=40, trainable=False),
        Bidirectional(GRU(32)),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m

def train_and_save():
    df = load_data()
    global tokenizer
    tokenizer, vocab_size = prepare_tokenizer(df['cleaned'])
    embedding_matrix, vocab_size = load_glove_embeddings('glove.6B.100d.txt', tokenizer, 100)

    seqs = tokenizer.texts_to_sequences(df['cleaned'])
    padded = pad_sequences(seqs, maxlen=40)
    Xtr, Xte, ytr, yte = train_test_split(padded, df['binary'], test_size=0.2, random_state=42)

    m = build_model(vocab_size, embedding_matrix)
    m.fit(Xtr, ytr, epochs=10, batch_size=16,
          validation_data=(Xte, yte), callbacks=[EarlyStopping(patience=2)], verbose=2)
    m.save('fake_news_model.h5')

# ---------- MODEL LOADER ----------
def load_resources():
    global model, tokenizer
    model = load_model('fake_news_model.h5')
    tokenizer = joblib.load('tokenizer.joblib')

# ---------- FLASK ----------
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', headlines=[], prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text'].strip()
    logging.debug(f"Input title: {text}")

    rss_url = f"https://news.google.com/rss/search?q={quote_plus(text)}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    for entry in feed.entries[:5]:
        title = entry.title
        score = SequenceMatcher(None, text.lower(), title.lower()).ratio()
        logging.debug(f"RSS candidate: {title} (sim={score:.2f})")
        if score > 0.9:
            logging.debug("RSS override: REAL")
            return render_template('index.html', headlines=[], prediction={'label':'REAL','real_pct':100.0,'fake_pct':0.0})

    urls = re.findall(r'https?://[^\s]+', text)
    if urls:
        logging.debug("URL override: REAL")
        return render_template('index.html', headlines=[], prediction={'label':'REAL','real_pct':100.0,'fake_pct':0.0})

    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=40)
    p = float(model.predict(pad)[0][0])
    label = 'REAL' if p > 0.5 else 'FAKE'
    real_pct = round(p * 100, 1)
    fake_pct = round((1 - p) * 100, 1)
    logging.debug(f"ML fallback - p: {p}, label: {label}")

    return render_template('index.html', headlines=[], prediction={'label':label,'real_pct':real_pct,'fake_pct':fake_pct})

# ---------- BOOTSTRAP ----------
if __name__ == '__main__':
    if not os.path.exists('fake_news_model.h5'):
        train_and_save()
    load_resources()
    app.run(debug=True)
