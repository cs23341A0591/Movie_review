import os
import re
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "tfidf.pkl"

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="IMDB Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
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


@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.drop_duplicates().reset_index(drop=True)
    df["clean_review"] = df["review"].apply(clean_text)
    df["word_count"] = df["clean_review"].apply(lambda x: len(x.split()))
    return df


@st.cache_data(show_spinner=False)
def top_words(text_series: pd.Series, n: int = 20):
    words = " ".join(text_series).split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return Counter(words).most_common(n)


@st.cache_resource(show_spinner=False)
def load_pretrained():
    """Load a pre-trained model + vectorizer from disk, if present."""
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    return None, None


@st.cache_resource(show_spinner="Training model...")
def train_model(reviews: pd.Series, labels: pd.Series, model_name: str, max_features: int):
    X_train, X_test, y_train, y_test = train_test_split(
        reviews, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = MultinomialNB()

    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds, labels=model.classes_)

    return model, tfidf, acc, report, cm, y_test, preds


def predict_sentiment(text: str, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else None
    confidence = max(proba) if proba is not None else None
    return pred, confidence


# --------------------------------------------------------------------------
# Sidebar - data + model controls
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader("Upload IMDB_Dataset.csv", type="csv")
default_path = "IMDB_Dataset.csv"

data_source = uploaded_file if uploaded_file is not None else default_path

sample_size = st.sidebar.slider(
    "Rows to use for training (smaller = faster)",
    min_value=1000,
    max_value=50000,
    value=8000,
    step=1000,
    help="The full dataset has ~50,000 reviews. Training on a subset is much faster for a live demo.",
)

pretrained_model, pretrained_vectorizer = load_pretrained()

use_pretrained = st.sidebar.checkbox(
    "Use pre-trained model (best_model.pkl)",
    value=pretrained_model is not None,
    disabled=pretrained_model is None,
    help="Loads best_model.pkl / tfidf.pkl (trained on the full dataset, ~89% accuracy) "
         "instead of retraining live. Uncheck to train fresh on the sample below.",
)
if pretrained_model is None:
    st.sidebar.caption("No best_model.pkl / tfidf.pkl found next to app.py — will train live.")

model_choice = st.sidebar.selectbox(
    "Model (used only when training live)",
    ["Logistic Regression", "Multinomial Naive Bayes"],
    disabled=use_pretrained,
)

max_features = st.sidebar.slider(
    "TF-IDF max features (used only when training live)",
    1000, 20000, 8000, step=1000, disabled=use_pretrained,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: increase the sample size and max features once you're ready for a more "
    "accurate (but slower) run."
)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write(
    "Explore audience sentiment from movie reviews, then train an NLP model "
    "to classify new reviews as **positive** or **negative**."
)

try:
    df_full = load_data(data_source)
except FileNotFoundError:
    st.error(
        "Couldn't find `IMDB_Dataset.csv` next to `app.py`. "
        "Upload the CSV using the sidebar, or place the file in the same folder as this app."
    )
    st.stop()

# Use a stratified sample for speed, but keep it representative
if sample_size < len(df_full):
    df, _ = train_test_split(
        df_full, train_size=sample_size, random_state=42, stratify=df_full["sentiment"]
    )
else:
    df = df_full

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🔎 Word Analysis", "🤖 Model & Evaluation", "✍️ Try It Yourself"]
)

# --------------------------------------------------------------------------
# Tab 1: Overview
# --------------------------------------------------------------------------
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total reviews (full dataset)", f"{len(df_full):,}")
    col2.metric("Reviews used for this run", f"{len(df):,}")
    col3.metric(
        "Positive share",
        f"{(df_full['sentiment'] == 'positive').mean() * 100:.1f}%",
    )

    st.subheader("Sample reviews")
    st.dataframe(df_full[["review", "sentiment"]].head(5), use_container_width=True)

    st.subheader("Sentiment distribution (full dataset)")
    sentiment_counts = df_full["sentiment"].value_counts()

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2", ax=ax)
        ax.set_ylabel("Count")
        ax.set_title("Reviews by Sentiment")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            colors=sns.color_palette("Set2"),
            startangle=90,
        )
        ax.set_title("Overall Audience Sentiment Share")
        st.pyplot(fig)

    st.subheader("Review length by sentiment")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x="word_count", hue="sentiment", bins=50, kde=True, palette="Set2", ax=ax)
    ax.set_xlim(0, 600)
    st.pyplot(fig)

# --------------------------------------------------------------------------
# Tab 2: Word Analysis
# --------------------------------------------------------------------------
with tab2:
    st.subheader("Most frequent words by sentiment")
    st.caption("Stop words removed. Based on the sampled rows selected in the sidebar.")

    n_words = st.slider("Number of top words to show", 5, 30, 15)

    pos_top = top_words(df.loc[df["sentiment"] == "positive", "clean_review"], n_words)
    neg_top = top_words(df.loc[df["sentiment"] == "negative", "clean_review"], n_words)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.barplot(x=[c for _, c in pos_top], y=[w for w, _ in pos_top], color="seagreen", ax=ax)
        ax.set_title("Top Words — Positive Reviews")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.barplot(x=[c for _, c in neg_top], y=[w for w, _ in neg_top], color="indianred", ax=ax)
        ax.set_title("Top Words — Negative Reviews")
        st.pyplot(fig)

# --------------------------------------------------------------------------
# Tab 3: Model & Evaluation
# --------------------------------------------------------------------------
with tab3:
    if use_pretrained:
        st.subheader("Using pre-trained model")
        model, tfidf = pretrained_model, pretrained_vectorizer

        # Evaluate the pre-trained model on a holdout split of the current sample,
        # purely for illustration — the model itself was already trained on the
        # full dataset, so treat this as a sanity check rather than a true holdout.
        _, X_eval, _, y_eval = train_test_split(
            df["clean_review"], df["sentiment"], test_size=0.3, random_state=7, stratify=df["sentiment"]
        )
        preds = model.predict(tfidf.transform(X_eval))
        acc = accuracy_score(y_eval, preds)
        report = classification_report(y_eval, preds, output_dict=True)
        cm = confusion_matrix(y_eval, preds, labels=model.classes_)
        y_test = y_eval

        st.info(
            "This model was pre-trained on the full dataset, so these numbers are a "
            "reference check on a sample split rather than a clean holdout."
        )
    else:
        st.subheader(f"Training live: {model_choice}")
        model, tfidf, acc, report, cm, y_test, preds = train_model(
            df["clean_review"], df["sentiment"], model_choice, max_features
        )

    st.success(f"Accuracy: **{acc * 100:.2f}%**")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Classification report**")
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    with c2:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    if isinstance(model, LogisticRegression):
        st.markdown("**Words most predictive of each sentiment**")
        feature_names = np.array(tfidf.get_feature_names_out())
        coefs = model.coef_[0]

        top_pos_idx = np.argsort(coefs)[-15:]
        top_neg_idx = np.argsort(coefs)[:15]

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.barh(feature_names[top_pos_idx], coefs[top_pos_idx], color="seagreen")
            ax.set_title("Drives POSITIVE prediction")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.barh(feature_names[top_neg_idx], coefs[top_neg_idx], color="indianred")
            ax.set_title("Drives NEGATIVE prediction")
            st.pyplot(fig)

# --------------------------------------------------------------------------
# Tab 4: Try it yourself
# --------------------------------------------------------------------------
with tab4:
    st.subheader("Predict sentiment for your own review")
    label = "the pre-trained model" if use_pretrained else f"the live-trained {model_choice} model"
    st.caption(f"Using {label}.")

    example = "This movie was absolutely fantastic, the acting and story were brilliant!"
    user_text = st.text_area("Enter a movie review:", value=example, height=150)

    if st.button("Predict sentiment", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some review text.")
        else:
            if use_pretrained:
                model, tfidf = pretrained_model, pretrained_vectorizer
            else:
                model, tfidf, *_ = train_model(df["clean_review"], df["sentiment"], model_choice, max_features)
            pred, confidence = predict_sentiment(user_text, model, tfidf)

            if pred == "positive":
                st.success(f"Predicted sentiment: **{pred.upper()}** 😀")
            else:
                st.error(f"Predicted sentiment: **{pred.upper()}** 🙁")

            if confidence is not None:
                st.progress(float(confidence))
                st.caption(f"Confidence: {confidence * 100:.1f}%")

st.markdown("---")
st.caption("Built with Streamlit, scikit-learn (TF-IDF + classical ML), and the IMDB 50K review dataset.")
