"""
streamlit_app.py
Ye Streamlit frontend hai. User image upload karega,
ye script us image ko FastAPI backend (/predict) ko bhejegi
aur predicted digit dikhayegi.

Run karne ka tareeqa:
    streamlit run streamlit_app.py

NOTE: Isse pehle api.py ko alag terminal me run karna zaroori hai:
    uvicorn api:app --reload --port 8000
"""

import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="MNIST Digit Predictor", layout="wide")

col1, col2 = st.columns(2)

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Digit ki image upload karein (png/jpg)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)

with col2:
    st.header("Prediction Result")

    if uploaded_file is not None:
        if st.button("Predict Digit"):
            with st.spinner("Prediction ho rahi hai..."):
                # File ko API ko bhejo
                files = {"file": uploaded_file.getvalue()}
                try:
                    response = requests.post(API_URL, files={"file": uploaded_file.getvalue()})
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Prediction Complete!")
                        st.subheader("Predicted Digit")
                        st.markdown(f"## {result['predicted_digit']}")
                        st.write(f"Confidence: {result['confidence']}%")
                    else:
                        st.error(f"API Error: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("API se connect nahi ho saka. Pehle 'uvicorn api:app --reload' chalayein.")
    else:
        st.info("Pehle koi image upload karein.")
