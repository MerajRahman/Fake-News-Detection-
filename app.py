import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import pickle

# ---------------- NLTK ---------------- #
@st.cache_resource
def download_nltk():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')  # যুক্ত করা হয়েছে (LookupError ফিক্স করার জন্য)
    nltk.download('wordnet')

download_nltk()

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_all():
    # সরাসরি .keras মডেল লোড করা হচ্ছে
    model = load_model("Sentiment_model_.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

model, tokenizer = load_all()

# ---------------- PREPROCESS ---------------- #
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

# ---------------- PREDICT ---------------- #
def predict_news(text):
    processed = preprocess_text(text)

    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=120, padding='post', truncating='post')

    prediction = model.predict(padded)

    class_index = np.argmax(prediction[0])
    confidence = float(prediction[0][class_index])

    label_map = {0: "Fake", 1: "Real"}
    return label_map[class_index], confidence

# ---------------- UI ---------------- #
st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("🔎 Verity Lens - Fake News Detector")

news_text = st.text_area("Enter News Text", height=200)

if st.button("Analyze"):
    if news_text.strip() == "":
        st.warning("Please enter some text")
    else:
        with st.spinner("Analyzing..."):
            label, confidence = predict_news(news_text)

        if label == "Fake":
            st.error(f"🚨 Fake News ({confidence:.2%} confidence)")
        else:
            st.success(f"✅ Real News ({confidence:.2%} confidence)")