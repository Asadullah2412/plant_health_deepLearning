import google.generativeai as genai

import os 

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# model = genai.GenerativeModel("models/gemini-2.0-flash")
model = genai.GenerativeModel("models/gemini-2.5-flash-lite")


def generate_explanation(disease,confidence):
    prompt = f"""
    You are an agricultural assistant.

    A plant disease detection model predicted the following:

    Disease: {disease}
    Confidence: {confidence}

    Instructions:
    - Only explain the disease mentioned above.
    - Do NOT introduce other diseases.
    - Use simple language suitable for farmers.
    - Keep the answer short and practical.

    Start the response with:

    Plant Health Report

    Detected Issue: {disease}
    Confidence: {round(confidence*100)}%

    Then provide:

    Cause:
    (1-2 short sentences)

    Treatment:
    - step 1
    - step 2
    - step 3

    Prevention:
    - tip 1
    - tip 2

    Do not include phrases like:
    "Here is an explanation"
    "As an AI model"
    """

    response = model.generate_content(prompt)

    return response.text