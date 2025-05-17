import streamlit as st
import joblib
import eli5
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Veracity Vigilance (Fake News Detection)")
st.markdown("Enter a news article or headline:")

user_input = st.text_area("News Text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        with st.spinner("Analyzing..."):
            vectorized_input = vectorizer.transform([user_input])
            prediction = model.predict(vectorized_input)[0]
            prob = model.predict_proba(vectorized_input)[0][prediction]

            if prediction == 1:
                st.success(f"✅ This news is **Real** with {prob*100:.2f}% confidence.")
            else:
                st.error(f"❌ This news is **Fake** with {prob*100:.2f}% confidence.")

            # Explain prediction using ELI5
            st.markdown("---")
            st.subheader("🧐 Why this prediction?")
            try:
                explanation = eli5.explain_prediction(model, user_input, vec=vectorizer)
                explanation_html = eli5.format_as_html(explanation)

                custom_css = """
                <style>
                    .eli5-weights tr, .eli5-weights td, .eli5-weights th {
                        color:black !important;
                        background-color: white !important;
                        font-size: 16px !important;
                    }
                    .eli5-weights th {
                        font-weight: bold !important;
                    }
                </style>
                """
                components.html(custom_css + explanation_html, height=500, scrolling=True)
            except Exception as e:
                st.warning(f"No explanation available for this prediction.\n\n{e}")
