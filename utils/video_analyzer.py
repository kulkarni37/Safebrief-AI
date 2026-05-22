import os
import time
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")


def summarize_video(video_file):
    """Analyze an uploaded video for safety incidents using Gemini native video support."""

    temp_path = None

    try:
        # Save uploaded video to a temp file
        suffix = ".mp4"
        if hasattr(video_file, "name"):
            ext = os.path.splitext(video_file.name)[1]
            if ext:
                suffix = ext

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_file.read())
            temp_path = tmp.name

        # Reset the file pointer for Streamlit to still be able to display it
        video_file.seek(0)

        # Upload video to Gemini File API
        print("[video_analyzer] Uploading video to Gemini...")
        uploaded_file = genai.upload_file(temp_path, mime_type=f"video/{suffix.lstrip('.')}")

        # Wait for video processing to complete
        print("[video_analyzer] Waiting for video processing...")
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(3)
            uploaded_file = genai.get_file(uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise Exception("Gemini video processing failed.")

        print("[video_analyzer] Video processed. Generating analysis...")

        prompt = """
You are an expert industrial safety inspector AI analyzing a workplace safety video.

Analyze this video carefully and provide a detailed safety assessment.

Respond STRICTLY in this format:

## Video Safety Summary
Describe what is happening in the video (2-3 sentences).

## Detected Safety Issues
• issue 1
• issue 2
• issue 3

## Risk Level
State the overall risk level (LOW / MEDIUM / HIGH) with brief explanation.

## Timeline of Events
• [timestamp/moment] — event description
• [timestamp/moment] — event description

## Recommended Actions
• action 1
• action 2
• action 3
• action 4

Be specific about what you observe. If the video does not show an industrial scene, still analyze for any general safety concerns.
"""

        response = model.generate_content([prompt, uploaded_file])

        # Clean up the uploaded file from Gemini
        try:
            genai.delete_file(uploaded_file.name)
        except Exception:
            pass

        return response.text

    except Exception as e:

        return f"""
## AI Video Safety Analysis

⚠️ **Video analysis could not be completed.**

**Error:** {str(e)}

**Possible fixes:**
• Check your GEMINI_API_KEY in `.env`
• Verify internet connection
• Ensure the video file is valid (MP4/MOV/AVI)
• Try a shorter video (under 2 minutes recommended)
"""

    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass