# eval_accuracy.py
import pandas as pd
import re
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report

# Load stopwords (run once if needed)
import nltk
nltk.download('stopwords', quiet=True)
STOP = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(w for w in text.split() if w not in STOP)

# 1. Load test data
cols = [
    'id','label','statement','subject','speaker','job_title','state','party',
    'barely_true','false','half_true','mostly_true','pants_on_fire','context'
]
df = pd.read_csv('test.tsv', sep='\t', names=cols)
label_map = {'true':1,'mostly-true':1,'half-true':1,'barely-true':0,'false':0,'pants-fire':0}
df = df.dropna(subset=['statement'])
y_true = df['label'].map(label_map).values

# 2. Preprocess
df['cleaned'] = df['statement'].apply(clean_text)
tokenizer = joblib.load('tokenizer.joblib')
seqs = tokenizer.texts_to_sequences(df['cleaned'])
X_test = pad_sequences(seqs, maxlen=40)

# 3. Load model and predict
model = load_model('fake_news_model.h5')
probs = model.predict(X_test, verbose=0).flatten()
y_pred = (probs > 0.5).astype(int)

# 4. Metrics
acc = accuracy_score(y_true, y_pred)
print(f"Test accuracy: {acc*100:.2f}%\n")
print(classification_report(y_true, y_pred, target_names=['FAKE','REAL']))
