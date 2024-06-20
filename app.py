import streamlit as st
from preprocessing import process_input
import joblib

MODEL_PATH = 'model.pkl'
model = joblib.load(MODEL_PATH)

# Streamlit App
st.set_page_config(page_title="Indonesian Clickbait Headline Detector", page_icon=":newspaper:", layout="centered")
st.title('Indonesian Clickbait Headline Detector')
user_input = st.text_input("Masukkan Judul Berita:")

preprocess_button = st.button('Preprocess')
if preprocess_button:
    if not user_input:
        st.write('Masukkan judul berita terlebih dahulu!')
        st.stop()
    processed_data = process_input(user_input)
    st.dataframe(processed_data)

predict_button = st.button('Predict')
if predict_button:
    if not user_input:
        st.write('Masukkan judul berita terlebih dahulu!')
        st.stop()
    processed_data = process_input(user_input)
    prediction = model.predict([processed_data])[0]
    result = 'Headline ini bukan clickbait' if prediction == 0 else 'Headline ini clickbait'
    st.write(result)
