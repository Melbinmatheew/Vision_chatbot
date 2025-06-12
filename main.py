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
    page_title="Visual AI Chat",
    page_icon="🤖",
    layout="wide"
)

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

def get_system_prompt(has_image, is_image_question):
    """Get appropriate system prompt based on context"""
    if has_image and is_image_question:
        return """You are an expert in reading and analyzing images. Answer questions based on what you can see in the image. 
        Be specific and detailed in your observations. If you cannot see something clearly, mention that."""
    else:
        return """You are a helpful AI assistant. Provide informative and accurate responses to user questions. 
        Be conversational and helpful."""

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
st.title("🤖 Visual AI Chat")
st.markdown("Upload an image and ask questions about it, or have a general conversation!")

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
    
    st.header("⚙️ Settings")
    max_tokens = st.slider("Max Response Length", 100, 1000, 500)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.header("💬 Chat")

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
    
    # Prepare messages for API call
    api_messages = []
    
    # Add system message
    system_prompt = get_system_prompt(has_image, is_image_question)
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
        with st.spinner("Thinking..."):
            try:
                # Choose model based on whether we're analyzing an image
                if has_image and is_image_question:
                    model = "opengvlab/internvl3-14b:free"
                else:
                    model = "opengvlab/internvl3-14b:free"
                
                completion = st.session_state.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "StreamlitVisualAIChat"
                    },
                    model=model,
                    messages=api_messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                
                response = completion.choices[0].message.content
                
                # Add context indicator
                if has_image and is_image_question:
                    context_note = "📸 *Analyzing uploaded image*\n\n"
                elif has_image and not is_image_question:
                    context_note = "💭 *General conversation (image available but not referenced)*\n\n"
                else:
                    context_note = "💬 *General conversation*\n\n"
                
                full_response = context_note + response
                st.write(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = str(e)
                st.error(f"Error generating response: {error_msg}")
                
                # Provide specific troubleshooting based on error
                if "422" in error_msg:
                    st.info("💡 **Troubleshooting Tips:**")
                    st.write("- Try re-uploading your image")
                    st.write("- Make sure your image file is not corrupted")
                    st.write("- Try a different image format (PNG, JPEG)")
                elif "401" in error_msg or "api key" in error_msg.lower():
                    st.info("🔑 Please check your OpenRouter API key in the .env file")
                else:
                    st.info("🔄 Please try again or contact support if the issue persists")

# ----------------------
# Information Panel
# ----------------------
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    ### 🖼️ Image Analysis Mode
    1. Upload an image using the sidebar
    2. Ask questions about the image (e.g., "What do you see?", "What's the area of room 154?")
    3. The app will analyze the image and provide detailed answers
    
    ### 💬 General Chat Mode
    1. Ask any general question
    2. The app will provide helpful responses without referencing the image
    3. You can switch between modes seamlessly
    
    ### 🔧 Tips
    - The app automatically detects if your question is about the uploaded image
    - Use specific keywords like "image", "picture", "see", "shown" for image analysis
    - Clear chat history when uploading a new image for better context
    - Adjust max response length in the sidebar
    """)

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("*Powered by OpenRouter API and Streamlit*")