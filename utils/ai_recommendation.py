import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def generate_recommendation(text):

    prompt = f"""
    You are an industrial safety expert.

    Analyze this incident and provide:

    1. Immediate actions
    2. Preventive measures
    3. Safety improvements
    4. Compliance recommendations

    Incident:
    {text}
    """

    response = model.generate_content(prompt)

    return response.text