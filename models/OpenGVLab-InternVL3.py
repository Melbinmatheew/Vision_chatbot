# import os
# import base64
# from dotenv import load_dotenv
# from openai import OpenAI


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
    
#     return f"data:{mime_type};base64,{base64_image}"


# # Load environment variables from .env file
# load_dotenv()

# # Retrieve values securely from environment variables
# api_key = os.getenv("openrouter_api_key")
# base_url = "https://openrouter.ai/api/v1"

# client = OpenAI(
#     base_url=base_url,
#     api_key=api_key,
# )

# # Convert image to base64 data URL
# image_path = r"D:\Melbin\SELF\openRouter\static\02_page_2.jpg"
# image_data_url = encode_image_to_base64(image_path)

# completion = client.chat.completions.create(
#   extra_headers={
#     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
#     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
#   },
#   extra_body={},
#   model="opengvlab/internvl3-14b:free",
#   messages=[
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "text",
#                 "text": "What is the area of room 154?"
#             },
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": image_data_url
#           }
#         }
#       ]
#     }
#   ]
# )
# print(completion.choices[0].message.content)

####=====================================================================================================================

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
        "role": "system",
        "content": """You are a document understanding assistant designed to analyze an input document and answer user questions specifically related to that document.

When a user provides a question and a document, carefully check if the question pertains only to the content of the provided document.

- If the question is unrelated to the document, first politely state that the question is outside the document's scope, then answer the question using your general knowledge.
- If the question relates to the document, analyze the relevant parts of the document thoroughly to produce an accurate, well-structured answer.

Additionally, when users submit questions or requests through chat involving documents or files, rewrite and refine their input text queries to improve clarity, correctness, and specificity. This rewriting step is crucial to enable precise document understanding and high-quality annotation extraction.

Your output must be structured clearly, highlighting:

- The refined user query (if rewriting was necessary).
- Whether the question is related to the document.
- A detailed, accurate answer (with references to the document where relevant).

Always encourage reasoning steps before providing conclusions. Ensure accuracy and clarity in all responses.

# Steps
1. Receive the user's question and the input document.
2. Determine if the question is about the document content.
3. If unrelated, notify the user and answer based on general knowledge.
4. If related, analyze the document and generate an accurate answer.
5. Rewrite the user's original question/request to improve clarity and precision.
6. Present output in a structured format as described.

# Output Format
Provide output as a JSON object with the following fields:
- "refined_query": the rewritten, polished user question.
- "is_question_related": boolean indicating if question relates to the document.
- "answer": the finally produced answer to the question, including references to the document if applicable.

Example:
{
  "refined_query": "What is the main cause of climate change according to the document?",
  "is_question_related": true,
  "answer": "According to the document, the main cause of climate change is greenhouse gas emissions due to fossil fuel combustion."
}

# Notes
- Focus on clarity and precision when rewriting user queries.
- If multiple documents or files are provided, consider all relevant information.
- Always separate the notice regarding question relevance from the answer.
- Maintain professional, helpful tone throughout.

This prompt guides you to perform refined document understanding, question relevance evaluation, query refinement, and structured output generation for accurate answers and annotations."""
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is machine learning?"
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

