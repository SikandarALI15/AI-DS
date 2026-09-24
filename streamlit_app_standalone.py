"""
streamlit_app_standalone.py
Ye version bina alag API/Render ke chalta hai.
Model seedha Streamlit app ke andar load hota hai.

Deploy karne ka tareeqa: Streamlit Community Cloud (share.streamlit.io)
Main file path: mnist-digit-classifier/streamlit_app_standalone.py
"""

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

st.set_page_config(page_title="MNIST Digit Predictor", layout="wide")

# Model ko ek baar load karo aur cache mein rakho (baar baar load na ho)
@st.cache_resource
def load_model():
    return keras.models.load_model("mnist_ann_model.h5")

model = load_model()


def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("L")             # grayscale
    img = img.resize((28, 28))           # MNIST size
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28)
    return img_array


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
                processed = preprocess_image(image)
                predictions = model.predict(processed)
                predicted_digit = int(np.argmax(predictions))
                confidence = float(np.max(predictions)) * 100

                st.success("Prediction Complete!")
                st.subheader("Predicted Digit")
                st.markdown(f"## {predicted_digit}")
                st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.info("Pehle koi image upload karein.")
