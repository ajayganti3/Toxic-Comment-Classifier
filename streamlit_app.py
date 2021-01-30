import streamlit as st
import warnings
import csv
import nltk
nltk.download('stopwords')

# Set the title of the page
st.set_page_config(page_title="Toxic Comment Classifier")

warnings.simplefilter(action="ignore")

st.title("Toxic Comment Classification")
st.subheader("Comment:")
comment = st.text_area(label="",max_chars=300)
btn = st.button(label="Predict")
st.subheader("Prediction:")
if btn:
    if comment != "":
        with open('comments.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([comment])
        progress = st.markdown("Loading...")
        from classifier import predict
        progress.markdown("Predicting...")
        output = predict(comment)
        progress.markdown(output)
    else:
        st.write("Please Enter Text")