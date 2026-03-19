import google.generativeai as genai
import json
from api import API_KEY

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("models/gemini-2.5-flash-lite")


def generate_explanation(disease, confidence):
    prompt = f"""
You are an agricultural assistant.

A plant disease detection model predicted:

Disease: {disease}
Confidence: {round(confidence*100)}%

Instructions:
- Only explain the given disease.
- Do NOT mention other diseases.
- Use simple, practical language for farmers.
- Keep responses concise and actionable.

Respond ONLY in valid JSON format:

{{
  "title": "Plant Health Report",
  "disease": "{disease}",
  "confidence": "{round(confidence*100)}%",
  "cause": "1-2 short sentences explaining the cause",
  "treatment": [
    "Step 1",
    "Step 2",
    "Step 3"
  ],
  "prevention": [
    "Tip 1",
    "Tip 2"
  ]
}}

Rules:
- No extra text outside JSON
- Do NOT wrap the response in markdown or code blocks
- No explanations before or after JSON
- Keep sentences short
"""

    response = model.generate_content(prompt)

    raw_text = response.text.strip()

    # 🔥 Remove markdown wrappers if present
    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw_text)
    except Exception as e:
        print("JSON parse error:", e)
        print("RAW RESPONSE:", raw_text)

        data = {
            "title": "Plant Health Report",
            "disease": disease,
            "confidence": f"{round(confidence*100)}%",
            "cause": "Unable to parse response",
            "treatment": [],
            "prevention": []
        }

    return data