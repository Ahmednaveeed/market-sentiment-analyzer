import streamlit as st
import pickle
from preprocess import preprocess_text

# Load the vectorizer, model, and label encoder
with open("Tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Title
st.title("📈 Tweet Sentiment Classifier")
st.markdown("Enter a tweet to classify it as **Bullish** or **Bearish**.")

# Input box
tweet = st.text_area("Tweet", "", placeholder="Like 'buy sol' or 'trump bans crypto'")

# Predict button
if st.button("Predict Sentiment"):
    if not tweet.strip():
        st.warning("Please enter a tweet.")
    else:
        # Preprocess
        cleaned = preprocess_text(tweet)
        # Turn into numbers
        vectorized = vectorizer.transform([cleaned])
        # Predict
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized).max()
        # Decode prediction
        label = label_encoder.inverse_transform([prediction])[0]
        # Output
        if label == 'bullish':
            st.success(f"📊 Sentiment: **Bullish** ({prob*100:.2f}% confidence)")
        else:
            st.error(f"📉 Sentiment: **Bearish** ({prob*100:.2f}% confidence)")