from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
import cv2
import base64

from llm_utils import generate_explanation
from model_utils import predict_image,is_leaf



app = FastAPI()

@app.get('/')
def home():
    return {"message": "Plant Disease Detection API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # STEP 1 → Leaf detection
    if not is_leaf(image):
        return {
            "prediction": "Invalid Image",
            "message": "Please upload a clear leaf image."
        }

    # STEP 2 → Disease detection
    label, confidence = predict_image(image)

    # STEP 3 → Confidence guard
    if confidence < 0.65:
        return {
            "prediction": "Uncertain",
            "message": "Image not clear. Please upload a clearer leaf photo."
        }

    # STEP 4 → LLM explanation
    explanation = generate_explanation(label, confidence)

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 1),
        "explanation": explanation
    }