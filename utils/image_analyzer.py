import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")


def analyze_image(image_file):
    """Analyze an uploaded image for industrial safety hazards using Gemini Vision."""

    try:
        # Convert uploaded file to PIL Image
        image = Image.open(image_file)

        prompt = """
You are an expert industrial safety inspector AI.

Analyze this image for workplace safety hazards and risks.

Respond STRICTLY in this format:

## Detected Hazards
• hazard 1
• hazard 2
• hazard 3

## Risk Assessment
State the overall risk level (LOW / MEDIUM / HIGH) with explanation.

## Possible Dangers
• danger 1
• danger 2
• danger 3

## Recommended Actions
• action 1
• action 2
• action 3
• action 4

Be specific about what you see in the image. If the image does not show an industrial or workplace scene, still analyze it for any general safety concerns.
"""

        response = model.generate_content([prompt, image])

        return response.text

    except Exception as e:

        return f"""
## AI Image Safety Analysis

⚠️ **Analysis could not be completed.**

**Error:** {str(e)}

**Possible fixes:**
• Check your GEMINI_API_KEY in `.env`
• Verify internet connection
• Ensure the image file is valid
• Try a different image format (PNG/JPG)
"""