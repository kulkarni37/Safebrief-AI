import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# NVIDIA CLIENT
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def generate_summary(text):

    try:

        # LIMIT HUGE PDF TEXT
        text = text[:5000]

        prompt = f"""
You are an industrial safety intelligence AI.

Analyze the industrial incident report.

Return the response STRICTLY in this format:

SUMMARY:
• point 1
• point 2
• point 3

ROOT CAUSE:
• point 1
• point 2

HAZARDS:
• point 1
• point 2
• point 3

RECOMMENDATIONS:
• point 1
• point 2
• point 3

Incident Report:
{text}
"""

        completion = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=700
        )

        result = completion.choices[0].message.content

        return result

    except Exception as e:

        return f"""
SUMMARY:
• Unable to generate AI summary

ROOT CAUSE:
• LLM processing failed

HAZARDS:
• API issue detected

RECOMMENDATIONS:
• Check NVIDIA API key
• Verify internet connection
• Retry request

ERROR:
{str(e)}
"""