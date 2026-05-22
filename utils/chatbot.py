import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def chatbot_response(question, incident_text):
    message = (
        "You are an industrial safety AI assistant.\n\n"
        f"Incident Report:\n{incident_text}\n\n"
        f"User Question:\n{question}"
    )

    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {
                "role": "user",
                "content": message
            }
        ],
        temperature=0.2,
        max_tokens=300
    )

    return response.choices[0].message.content