import streamlit as st
from svcmodel import getPrediction, getMetrics, getData
st.title("Sentiment analysis with SVM")
st.write("Write your review to predict if it is positive or negative.")
# text entry
user_input = st.text_area("Review text", "")

if st.button("Predict"):
    if user_input:
        predicted_label = getPrediction(user_input)
        st.write(f"The review is predicted as : {predicted_label}")
    else:
        st.write("Write your review to get your prediction.")

Accuracy, Recall, F1Score, std =getMetrics()
st.write(f"Accuracy: {Accuracy:.2f}")
st.write(f"Recall: {Recall:.2f}")
st.write(f"F1 Score: {F1Score:.2f}")
st.write(f"Std: {std:.2f}")
data = getData()
st.write(data)