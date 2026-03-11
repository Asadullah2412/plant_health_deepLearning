import google.generativeai as genai

import os 

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model =genai.GenerativeModel('gemini-1.5-flash')


def generate_explanation(disease,confidence):
    prompt = f"""
    A plant disease detection AI predicted:

    Disease: {disease}
    Confidence: {confidence}

    Explain the disease in simple terms for farmers.
    Include:
    - what the disease is
    - why it happens
    - simple treatment steps
    """

    response = model.generate_content(prompt)

    return response.text