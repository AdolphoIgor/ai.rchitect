# pip install -U google-genai
import os
from dotenv import load_dotenv

load_dotenv()

from google import genai

client = genai.Client()

response = client.models.generate_content(
    # model="gemini-3-flash-preview",
    # model="gemini-2.5-flash-lite-preview-09-2025",
    # model="gemini-2.5-flash-lite",
    # model="gemini-2.5-flash-preview-09-2025",
    # model="gemini-2.5-flash",
    # model="gemini-3-flash-preview",
    model="gemini-2.0-flash",
    contents="Explain how AI works in a few words"
)

print(response.text)