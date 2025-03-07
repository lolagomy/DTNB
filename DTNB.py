import streamlit as st
import joblib
import numpy as np

vectorizer = joblib.load('tfidf_vectorizer.pkl')
nb_model = joblib.load('naive_bayes_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')


st.title("IMDb Movie Review Sentiment Analysis 🎬")
st.markdown("<h4 style = 'margin: -30px; color: #FFC470; text-align: center; font-family: Serif '>BUILT by LOLA</h4>", unsafe_allow_html = True)
st.write("Enter a movie review and choose a model to predict whether it's positive or negative.")

user_review = st.text_area("Enter your review:", "")

model_choice = st.selectbox("Choose a model:", ["Naive Bayes", "Decision Tree"])

if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review!")
    else:
        review_vectorized = vectorizer.transform([user_review])

        if model_choice == "Naive Bayes":
            prediction = nb_model.predict(review_vectorized)
        else:
            prediction = dt_model.predict(review_vectorized)

        sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😞"
        st.success(f"Predicted Sentiment: **{sentiment}**")
