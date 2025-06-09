import os
import base64
from dotenv import load_dotenv
from openai import OpenAI


def encode_image_to_base64(image_path):
    """Convert local image file to base64 data URL"""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Determine the MIME type based on file extension
    if image_path.lower().endswith(('.png', '.PNG')):
        mime_type = "image/png"
    elif image_path.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
        mime_type = "image/jpeg"
    elif image_path.lower().endswith(('.gif', '.GIF')):
        mime_type = "image/gif"
    elif image_path.lower().endswith(('.webp', '.WEBP')):
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"  # Default fallback
    
    return f"data:{mime_type};base64,{base64_image}"


# Load environment variables from .env file
load_dotenv()

# Retrieve values securely from environment variables
api_key = os.getenv("openrouter_api_key")
base_url = "https://openrouter.ai/api/v1"

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

# Convert image to base64 data URL
image_path = r"D:\Melbin\SELF\openRouter\static\02_page_2.jpg"
image_data_url = encode_image_to_base64(image_path)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  extra_body={},
  model="moonshotai/kimi-vl-a3b-thinking:free",
  messages=[
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is the area of room 218?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url
          }
        }
      ]
    }
  ]
)
print(completion.choices[0].message.content)