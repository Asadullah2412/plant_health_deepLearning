from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
import cv2
import base64

from llm_utils import generate_explanation
from model_utils import predict_image



app = FastAPI()

@app.get('/')
def home():
    return {"message": "Plant Disease Detection API running"}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    label, confidence = predict_image(image)
   
    if confidence < 0.65:
        return {
            "prediction": "Uncertain",
            "message": "Image not clear. Please upload a clearer leaf photo."
        }

    explanation = generate_explanation(label, confidence)

    # convert image to numpy
    img_np = np.array(image)
    img_resized = cv2.resize(img_np, (224, 224))

    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 1),
        "explanation": explanation,
        
    }