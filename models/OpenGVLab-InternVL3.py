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
  model="opengvlab/internvl3-14b:free",
  messages=[
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is the area of room 154?"
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


####=====================================================================================================================

# ### OpenRouter API Example with MLflow Logging

# import os
# import base64
# from dotenv import load_dotenv
# from openai import OpenAI
# import requests
# import mlflow
# from datetime import datetime

# def encode_image_to_base64(image_path):
#     """Convert local image file to base64 data URL"""
#     with open(image_path, "rb") as image_file:
#         base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
#     # Determine the MIME type based on file extension
#     if image_path.lower().endswith(('.png', '.PNG')):
#         mime_type = "image/png"
#     elif image_path.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
#         mime_type = "image/jpeg"
#     elif image_path.lower().endswith(('.gif', '.GIF')):
#         mime_type = "image/gif"
#     elif image_path.lower().endswith(('.webp', '.WEBP')):
#         mime_type = "image/webp"
#     else:
#         mime_type = "image/jpeg"  # Default fallback
    
#     return f"data:{mime_type};base64,{base64_image}",mime_type


# # Load environment variables from .env file
# load_dotenv()
# api_key = os.getenv("openrouter_api_key")

# ## Image + prompt
# image_path = r"D:\Melbin\SELF\openRouter\static\02_page_2.jpg"
# prompt_text = "What is the area of room 218?" 
# model_name = "opengvlab/internvl3-14b:free"

# # Encode image to base64
# image_data_url, mime_type = encode_image_to_base64(image_path)

# ## Prepare the payload for the API request
# payload = {
#     "model": model_name,
#     "messages": [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": prompt_text},
#                 {"type": "image_url", "image_url": {"url": image_data_url}}
#             ]
#         }
#     ]
# }

# headers = {
#     "Authorization": f"Bearer {api_key}",
#     "Content-Type": "application/json",
#     # Optional headers for OpenRouter ranking (you can update or omit)
#     "HTTP-Referer": "<YOUR_SITE_URL>",
#     "X-Title": "<YOUR_SITE_NAME>"
# }

# # Set up MLflow
# mlflow.set_experiment("llm_visual_reasoning")

# with mlflow.start_run(run_name=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

#     # Log parameters
#     mlflow.log_param("model", model_name)
#     mlflow.log_param("prompt", prompt_text)
#     mlflow.log_param("image_path", os.path.basename(image_path))
#     mlflow.set_tags({
#         "task": "visual_question_answering",
#         "image_type": mime_type,
#         "model_provider": "openrouter"
#     })

#     # Log image as artifact
#     mlflow.log_artifact(image_path, artifact_path="input_image")

#     # Make API request
#     response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    
#     if response.status_code == 200:
#         result = response.json()
#         response_text = result["choices"][0]["message"]["content"]
        
#         # Log response
#         mlflow.log_text(response_text, "response.txt")
#         mlflow.log_metric("response_length", len(response_text))
        
#         print("\n📤 Response from model:\n", response_text)
#     else:
#         mlflow.log_param("status_code", response.status_code)
#         mlflow.log_param("error", str(response.text))
#         print("\n❌ Error:", response.text)

