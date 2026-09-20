"""
api.py
Ye FastAPI backend hai jo trained ANN model ko load karke
ek /predict endpoint provide karta hai. Isme aap image bhej kar
predicted digit hasil kar sakte hain.

Run karne ka tareeqa:
    uvicorn api:app --reload --port 8000

Test karne ka tareeqa:
    http://127.0.0.1:8000/docs  (Swagger UI)
"""

import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras

app = FastAPI(title="MNIST Digit Prediction API")

# Streamlit se request aane ke liye CORS allow karo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trained model load karo (train_model.py chalane ke baad ye file bani hogi)
model = keras.models.load_model("mnist_ann_model.h5")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Uploaded image ko model ke input format (1, 28, 28) me convert karta hai.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    img = img.resize((28, 28))                              # MNIST size
    img_array = np.array(img).astype("float32") / 255.0      # normalize
    img_array = img_array.reshape(1, 28, 28)                 # batch dimension add karo
    return img_array


@app.get("/")
def home():
    return {"message": "MNIST Digit Prediction API chal rahi hai."}


@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)

    predictions = model.predict(processed_image)
    predicted_digit = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence * 100, 2)
    }
