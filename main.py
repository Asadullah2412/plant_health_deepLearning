from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from llm_utils import generate_explanation

from model_utils import predict_image

app = FastAPI()

@app.get('/')
def home():
    return {"message": "Plant Disease Detection API running"}

@app.post('/predict')
async def predict(file:UploadFile = File(...)):
    contents =await file.read()
    image = Image.open(io.BytesIO(contents))

    label ,confidence = predict_image(image)

    return{
        'prediction':label,
        'confidence': confidence
    }
