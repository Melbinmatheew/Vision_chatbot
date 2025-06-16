import streamlit as st
import os
import base64
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import io

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="Visual AI Chat - Free Models",
    page_icon="🤖",
    layout="wide"
)

# ----------------------
# Model Configuration - FREE MODELS ONLY
# ----------------------
VISION_MODELS = {
    "InternVL3-14B": {
        "id": "opengvlab/internvl3-14b:free",
        "description": "High-quality open-source vision model (Free)",
        "supports_vision": True
    },
    "Llava 1.5 7B": {
        "id": "liuhaotian/llava-yi-34b:free",
        "description": "Open-source vision-language model (Free)",
        "supports_vision": True
    },
    "Qwen2-VL 7B": {
        "id": "qwen/qwen-2-vl-7b-instruct:free",
        "description": "Alibaba's free vision-language model",
        "supports_vision": True
    }
}

TEXT_MODELS = {
    "Llama 3.1 8B": {
        "id": "meta-llama/llama-3.1-8b-instruct:free",
        "description": "Meta's free Llama model",
        "supports_vision": False
    },
    "Gemma 2 9B": {
        "id": "google/gemma-2-9b-it:free",
        "description": "Google's free Gemma model",
        "supports_vision": False
    },
    "Qwen2.5 7B": {
        "id": "qwen/qwen-2.5-7b-instruct:free",
        "description": "Alibaba's free text model",
        "supports_vision": False
    },
    "Mixtral 8x7B": {
        "id": "mistralai/mixtral-8x7b-instruct:free",
        "description": "Mistral's free mixture-of-experts model",
        "supports_vision": False
    }
}

# ----------------------
# Utility Functions
# ----------------------
def encode_image_to_base64(image_file):
    """Convert uploaded image to base64 string"""
    try:
        # Reset file pointer to beginning
        image_file.seek(0)
        # Read image bytes
        image_bytes = image_file.read()
        
        # Validate image data
        if not image_bytes:
            st.error("Image file is empty")
            return None
            
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Validate base64 encoding
        if not base64_image:
            st.error("Failed to encode image to base64")
            return None
        
        # MIME type inference
        if image_file.type == "image/png":
            mime_type = "image/png"
        elif image_file.type in ["image/jpeg", "image/jpg"]:
            mime_type = "image/jpeg"
        elif image_file.type == "image/gif":
            mime_type = "image/gif"
        elif image_file.type == "image/webp":
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"
        
        data_url = f"data:{mime_type};base64,{base64_image}"
        
        # Validate final data URL
        if len(data_url) < 50:  # Basic validation
            st.error("Generated data URL seems too short")
            return None
            
        return data_url
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return None

def initialize_client():
    """Initialize OpenAI client with OpenRouter"""
    load_dotenv()
    api_key = os.getenv("openrouter_api_key")
    
    if not api_key:
        st.error("OpenRouter API key not found. Please set 'openrouter_api_key' in your .env file.")
        return None
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

def is_image_related_question(question, has_image):
    """Determine if the question is related to the uploaded image"""
    if not has_image:
        return False
    
    # Keywords that suggest image-related questions
    image_keywords = [
        'image', 'picture', 'photo', 'see', 'shown', 'display', 'visual',
        'what is', 'what are', 'describe', 'identify', 'room', 'area',
        'text', 'read', 'written', 'color', 'shape', 'object', 'person',
        'building', 'document', 'chart', 'graph', 'table', 'diagram'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in image_keywords)

def get_system_prompt(has_image, is_image_question, custom_prompt=""):
    """Get appropriate system prompt based on context"""
    base_prompt = ""
    
    if has_image and is_image_question:
        base_prompt = """You are an expert in reading and analyzing images. Answer questions based on what you can see in the image. 
        Be specific and detailed in your observations. If you cannot see something clearly, mention that."""
    else:
        base_prompt = """You are a helpful AI assistant. Provide informative and accurate responses to user questions. 
        Be conversational and helpful."""
    
    # Add custom prompt if provided
    if custom_prompt.strip():
        base_prompt += f"\n\nAdditional instructions: {custom_prompt.strip()}"
    
    return base_prompt

def get_model_info(model_name, model_dict):
    """Get model information from the model dictionary"""
    return model_dict.get(model_name, {"id": model_name, "description": "Unknown model"})

# ----------------------
# Initialize Session State
# ----------------------
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'image_data_url' not in st.session_state:
    st.session_state.image_data_url = None
if 'client' not in st.session_state:
    st.session_state.client = initialize_client()

# ----------------------
# Main App Layout
# ----------------------
st.title("🤖 Visual AI Chat - Free Models")
st.markdown("Upload an image and ask questions about it, or have a general conversation! **All models are completely free to use.**")

# Add a banner highlighting free usage
st.success("🎉 **100% FREE** - All models in this app are free to use with your OpenRouter account!")

# Sidebar for image upload and settings
with st.sidebar:
    st.header("🖼️ Image Upload")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
        help="Upload an image to ask questions about it"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Encode image if it's new
        if st.session_state.current_image != uploaded_file.name:
            st.session_state.current_image = uploaded_file.name
            with st.spinner("Processing image..."):
                st.session_state.image_data_url = encode_image_to_base64(uploaded_file)
            
            if st.session_state.image_data_url:
                st.success("✅ Image uploaded successfully!")
            else:
                st.error("❌ Failed to process image")
                
        # Clear previous messages when new image is uploaded
        if st.button("🔄 New Image - Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    st.header("🤖 Model Selection")
    
    # Vision Model Selection
    st.subheader("🔍 Vision Models (FREE)")
    vision_model_names = list(VISION_MODELS.keys())
    selected_vision_model = st.selectbox(
        "Choose vision model:",
        vision_model_names,
        index=0,
        help="Select the AI model for analyzing images - all are free!"
    )
    
    # Show model description
    vision_model_info = VISION_MODELS[selected_vision_model]
    st.info(f"📝 {vision_model_info['description']}")
    
    # Text Model Selection
    st.subheader("💬 Text Models (FREE)")
    text_model_names = list(TEXT_MODELS.keys())
    selected_text_model = st.selectbox(
        "Choose text model:",
        text_model_names,
        index=0,
        help="Select the AI model for general conversation - all are free!"
    )
    
    # Show model description
    text_model_info = TEXT_MODELS[selected_text_model]
    st.info(f"📝 {text_model_info['description']}")
    
    st.header("⚙️ Settings")
    
    # Custom System Prompt
    st.subheader("Custom Instructions")
    custom_prompt = st.text_area(
        "Additional instructions for the AI:",
        placeholder="E.g., 'Always respond in a friendly tone', 'Focus on technical details', 'Be concise', etc.",
        help="Add custom instructions that will be included in every conversation",
        height=100
    )
    
    # Other settings
    max_tokens = st.slider("Max Response Length", 100, 2000, 800)
    temperature = st.slider("Creativity Level", 0.0, 1.0, 0.7, 0.1, 
                           help="Lower values = more focused, Higher values = more creative")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.header("💬 Chat")

# Display current model info
col1, col2 = st.columns(2)
with col1:
    st.info(f"🖼️ **Vision Model:** {selected_vision_model} 🆓")
with col2:
    st.info(f"💬 **Text Model:** {selected_text_model} 🆓")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the image or anything else..."):
    if not st.session_state.client:
        st.error("Please configure your OpenRouter API key first.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Determine if this is an image-related question
    has_image = st.session_state.image_data_url is not None
    is_image_question = is_image_related_question(prompt, has_image)
    
    # Validate image data if needed
    if has_image and is_image_question:
        if not st.session_state.image_data_url or len(st.session_state.image_data_url) < 50:
            st.error("Image data is invalid. Please re-upload your image.")
            st.stop()
    
    # Choose appropriate model
    if has_image and is_image_question:
        selected_model = vision_model_info["id"]
        model_type = "vision"
        model_name = selected_vision_model
    else:
        selected_model = text_model_info["id"]
        model_type = "text"
        model_name = selected_text_model
    
    # Prepare messages for API call
    api_messages = []
    
    # Add system message with custom prompt
    system_prompt = get_system_prompt(has_image, is_image_question, custom_prompt)
    api_messages.append({
        "role": "system",
        "content": system_prompt
    })
    
    # For image questions, create a simplified message structure
    if has_image and is_image_question:
        # Add only the current question with image
        api_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": st.session_state.image_data_url}}
            ]
        })
    else:
        # Add conversation history for text-only chat
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                api_messages.append({"role": "user", "content": msg["content"]})
            else:
                # Clean assistant messages (remove context indicators)
                clean_content = msg["content"]
                if clean_content.startswith("📸") or clean_content.startswith("💭") or clean_content.startswith("💬"):
                    # Remove the first line (context indicator)
                    clean_content = "\n".join(clean_content.split("\n")[2:])
                api_messages.append({"role": "assistant", "content": clean_content})
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {model_name} (Free)..."):
            try:
                completion = st.session_state.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "StreamlitVisualAIChat"
                    },
                    model=selected_model,
                    messages=api_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                
                response = completion.choices[0].message.content
                
                # Add context indicator
                if has_image and is_image_question:
                    context_note = f"📸 *Analyzing with {model_name} (Free)*\n\n"
                elif has_image and not is_image_question:
                    context_note = f"💭 *{model_name} (Free) - image available but not referenced*\n\n"
                else:
                    context_note = f"💬 *{model_name} (Free)*\n\n"
                
                full_response = context_note + response
                st.write(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = str(e)
                st.error(f"Error with {model_name}: {error_msg}")
                
                # Provide specific troubleshooting based on error
                if "422" in error_msg:
                    st.info("💡 **Troubleshooting Tips:**")
                    st.write("- Try re-uploading your image")
                    st.write("- Make sure your image file is not corrupted")
                    st.write("- Try a different image format (PNG, JPEG)")
                    st.write("- Try a different vision model")
                elif "401" in error_msg or "api key" in error_msg.lower():
                    st.info("🔑 Please check your OpenRouter API key in the .env file")
                elif "rate limit" in error_msg.lower():
                    st.info("⏱️ Rate limit reached. Please wait a moment and try again.")
                else:
                    st.info("🔄 Please try again with a different model or contact support if the issue persists")

# ----------------------
# Information Panel
# ----------------------
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    ### 🆓 **100% Free AI Models**
    All models in this app are completely free to use with your OpenRouter account!
    
    ### 🖼️ Image Analysis Mode
    1. Upload an image using the sidebar
    2. Select your preferred **Vision Model** from the dropdown (all free!)
    3. Ask questions about the image (e.g., "What do you see?", "Describe this image")
    4. The app will automatically use the vision model for image analysis
    
    ### 💬 General Chat Mode
    1. Select your preferred **Text Model** from the dropdown (all free!)
    2. Ask any general question
    3. The app will use the text model for general conversation
    4. You can switch between modes seamlessly
    
    ### 🎛️ Customization Options
    - **Model Selection**: Choose different free AI models for different tasks
    - **Custom Instructions**: Add your own instructions to guide the AI's behavior
    - **Temperature**: Adjust creativity level (0 = focused, 1 = creative)
    - **Max Tokens**: Control response length
    
    ### 🔧 Tips
    - **All Free**: No credits required - just your OpenRouter API key
    - **Model Performance**: Different models excel at different tasks
    - **Custom Prompts**: Use custom instructions for specific use cases
    - **Clear Chat**: Start fresh when switching contexts or uploading new images
    
    ### 🤖 Available Free Models
    **Vision Models** (for image analysis):
    - InternVL3-14B - High-quality open-source vision model
    - Llava 1.5 7B - Open-source vision-language model
    - Qwen2-VL 7B - Alibaba's vision-language model
    
    **Text Models** (for general chat):
    - Llama 3.1 8B - Meta's efficient language model
    - Gemma 2 9B - Google's optimized model
    - Qwen2.5 7B - Alibaba's latest text model
    - Mixtral 8x7B - Mistral's mixture-of-experts model
    """)

# Model comparison section
with st.expander("📊 Free Model Comparison & Tips"):
    st.markdown("""
    ### 🏆 Model Strengths (All Free!)
    
    **For Image Analysis:**
    - **InternVL3-14B**: Best overall vision capabilities, excellent for detailed analysis
    - **Llava 1.5 7B**: Good general-purpose vision model, fast responses
    - **Qwen2-VL 7B**: Strong at text extraction and document analysis
    
    **For Text Conversations:**
    - **Llama 3.1 8B**: Great reasoning and problem-solving
    - **Gemma 2 9B**: Excellent for factual questions and explanations
    - **Qwen2.5 7B**: Good balance of speed and quality
    - **Mixtral 8x7B**: Best for complex reasoning and creative tasks
    
    ### 💡 Custom Prompt Examples
    - **For Technical Analysis**: "Focus on technical details and provide precise measurements when possible"
    - **For Creative Description**: "Use vivid, descriptive language and focus on artistic elements"
    - **For Educational Content**: "Explain concepts simply and provide learning-focused insights"
    - **For Professional Use**: "Maintain a formal tone and focus on business-relevant information"
    
    ### 🎯 Best Practices for Free Models
    - **Be Specific**: Clear, specific questions get better responses
    - **Try Different Models**: Each model has different strengths
    - **Use Custom Instructions**: Tailor responses to your needs
    - **Experiment**: Free models let you experiment without cost!
    """)

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("*🆓 Powered by Free OpenRouter Models • 100% No Cost to Use • Streamlit Framework*")
st.markdown("**Pro Tip:** Create your free OpenRouter account at [openrouter.ai](https://openrouter.ai) to get started!")