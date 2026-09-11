import os
import re
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "tfidf.pkl"
DATA_PATH = "IMDB_Dataset.csv"

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)

sns.set_style("whitegrid")

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner=False)
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def top_words(_df, sentiment, n=10):
    text = " ".join(_df.loc[_df["sentiment"] == sentiment, "review"].apply(clean_text))
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return Counter(words).most_common(n)


def predict_sentiment(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else None
    confidence = max(proba) if proba is not None else None
    return pred, confidence


# --------------------------------------------------------------------------
# Load model (required) and data (optional, for the insights section)
# --------------------------------------------------------------------------
if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    st.error(
        "Missing `best_model.pkl` or `tfidf.pkl`. Place both files in the same "
        "folder as `app.py` and reload the page."
    )
    st.stop()

model, vectorizer = load_model()
df = load_data()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Paste in a movie review and instantly see whether it reads as positive or negative.")

st.markdown("---")

# --------------------------------------------------------------------------
# Main: predict
# --------------------------------------------------------------------------
user_text = st.text_area(
    "Your review",
    placeholder="e.g. The acting was incredible and the story kept me hooked till the end...",
    height=160,
    label_visibility="collapsed",
)

predict_clicked = st.button("Analyze Sentiment", type="primary", use_container_width=True)

if predict_clicked:
    if not user_text.strip():
        st.warning("Please enter a review first.")
    else:
        pred, confidence = predict_sentiment(user_text, model, vectorizer)

        if pred == "positive":
            st.success("### 😀 Positive")
        else:
            st.error("### 🙁 Negative")

        if confidence is not None:
            st.write(f"Confidence: **{confidence * 100:.1f}%**")
            st.progress(float(confidence))

st.markdown("---")

# --------------------------------------------------------------------------
# Optional: quick dataset insights (only if the CSV is available)
# --------------------------------------------------------------------------
if df is not None:
    with st.expander("📊 See audience sentiment insights from the dataset"):
        counts = df["sentiment"].value_counts()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total reviews", f"{len(df):,}")
        c2.metric("Positive", f"{(counts.get('positive', 0) / len(df)) * 100:.1f}%")
        c3.metric("Negative", f"{(counts.get('negative', 0) / len(df)) * 100:.1f}%")

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.pie(
            counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=sns.color_palette("Set2"), startangle=90,
        )
        st.pyplot(fig)

        st.markdown("**Most common words**")
        sample_df = df.sample(min(len(df), 3000), random_state=42)
        pos_top = top_words(sample_df, "positive")
        neg_top = top_words(sample_df, "negative")

        wc1, wc2 = st.columns(2)
        with wc1:
            st.caption("In positive reviews")
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.barplot(x=[c for _, c in pos_top], y=[w for w, _ in pos_top], color="seagreen", ax=ax)
            st.pyplot(fig)
        with wc2:
            st.caption("In negative reviews")
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.barplot(x=[c for _, c in neg_top], y=[w for w, _ in neg_top], color="indianred", ax=ax)
            st.pyplot(fig)

st.caption("Model: Logistic Regression + TF-IDF, trained on the IMDB 50K review dataset (~89% accuracy).")
