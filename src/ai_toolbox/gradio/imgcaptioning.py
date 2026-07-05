import sys
import os
import base64
import requests
import warnings
import io
from PIL import Image  # Native Python library to manage image scaling

# 1. Dynamically find and load your master .env configurations
flask_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'genai_flask_app'))
if flask_app_path not in sys.path:
    sys.path.append(flask_app_path)

from config import Config
from model import llm  # Your active master ChatLiteLLM instance
from langchain_core.messages import HumanMessage

# Suppress framework layout and package version warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
warnings.filterwarnings('ignore')

# 2. Network Downloader and Base64 Encoder

import base64
import requests
import io
from PIL import Image

def encode_images_to_base64(image_urls):
    """
    Downloads, compresses, and encodes a list of image URLs to base64 strings.
    Flattens RGBA transparency layers to RGB to prevent JPEG compilation errors.
    """
    encoded_images = []
    for url in image_urls:
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                # 1. Load the raw network data byte stream into Pillow
                img_data = io.BytesIO(response.content)
                img = Image.open(img_data)
                
                # FIX: Handle transparent RGBA images gracefully by converting to RGB [1]
                if img.mode in ("RGBA", "P"):
                    # Create a solid white background matching the image size [1]
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    # Paste the image on top of the white background using its alpha channel as a mask [1]
                    background.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None) # [1]
                    img = background
                elif img.mode != "RGB":
                    # For any other strange modes (like grayscale/CMYK), convert directly [1]
                    img = img.convert("RGB") # [1]
                
                # 2. Downscale the image if it is excessively large
                max_dimension = 1024
                if max(img.size) > max_dimension:
                    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                
                # 3. Compress the image bytes into a temporary in-memory buffer using JPEG formatting
                compressed_buffer = io.BytesIO()
                img.save(compressed_buffer, format="JPEG", quality=85) # [1]
                
                # 4. Read the optimized compressed bytes and encode to base64
                optimized_bytes = compressed_buffer.getvalue()
                encoded_image = base64.b64encode(optimized_bytes).decode("utf-8")
                
                mb_size = len(optimized_bytes) / (1024 * 1024)
                print(f"📥 Encoded and compressed asset: {url[:45]}... (Safe Size: {mb_size:.2f} MB)")
                
                encoded_images.append(encoded_image)
            else:
                print(f"⚠️ Failed to fetch image from {url} (Status code: {response.status_code})")
                encoded_images.append(None)
        except Exception as e:
            print(f"❌ Fault downscaling or downloading image file: {e}")
            encoded_images.append(None)
            
    return encoded_images



# 3. Modernized Multimodal LLM Handler (Replaces legacy model.chat)
def generate_model_response(encoded_image, user_query, assistant_prompt="You are a helpful assistant. Answer the following user query in 1 or 2 sentences: "):
    """
    Sends an image and a query to your active model using native LangChain HumanMessages.
    """
    if not encoded_image:
        return "Error: Image data is empty or invalid."

    # FIX: Assemble modern multimodal payloads natively
    # We pass the prompt + query as text, and attach the image bytes directly
    message_content = [
        {
            "type": "text", 
            "text": f"{assistant_prompt}\n\nUser Query: {user_query}"
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        }
    ]
    
    # Wrap your content array directly inside a native LangChain HumanMessage
    msg = HumanMessage(content=message_content)
    
    # Apply your specific low-temperature overrides natively using .bind()
    # This matches your course's TextChatParameters requirement safely
    vision_runner = llm.bind(temperature=0.2, max_tokens=100)
    
    # Invoke the model and extract the plain text string instantly via .content
    response = vision_runner.invoke([msg])
    return response.content.strip()

# ------------------------------------------------------------------
# TESTING INTERFACES & EXECUTION PIPELINE
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Sample Cloud URL links from your dataset lab documentation
    url_image_1 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5uo16pKhdB1f2Vz7H8Utkg/image-1.png'
    url_image_2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fsuegY1q_OxKIxNhf6zeYg/image-2.png'
    url_image_3 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/KCh_pM9BVWq_ZdzIBIA9Fw/image-3.png'
    url_image_4 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VaaYLw52RaykwrE3jpFv7g/image-4.png'

    image_urls = [url_image_1, url_image_2, url_image_3, url_image_4] 
    
    print("⏳ Downloading and caching data assets...")
    encoded_images = encode_images_to_base64(image_urls)
    
    print(f"\n🚀 Running Multimodal Vision Processing targeting: {Config.LLM_PROVIDER.upper()}")
    print("=" * 60)
    
    # Turn 1: Batch Description Loops
    user_query = "Describe the photo"
    for i, image in enumerate(encoded_images):
        if image:
            response = generate_model_response(image, user_query)
            print(f"📝 Description for image {i + 1}: {response}\n")

    print("=" * 60)
    
    # Turn 2: Specific Car Counting Verification Task
    if len(encoded_images) > 1 and encoded_images[1]:
        q2 = "How many cars are in this image?"
        print("👤 User Query: ", q2)
        print("🤖 Model Response: ", generate_model_response(encoded_images[1], q2))
        print("-" * 40)

    # Turn 3: Damage Severity Classification Task
    if len(encoded_images) > 2 and encoded_images[2]:
        q3 = "How severe is the damage in this image?"
        print("👤 User Query: ", q3)
        print("🤖 Model Response: ", generate_model_response(encoded_images[2], q3))
        print("-" * 40)

    # Turn 4: Nutrition Label Information Parsing Extraction Task
    if len(encoded_images) > 3 and encoded_images[3]:
        q4 = "How much sodium is in this product?"
        print("👤 User Query: ", q4)
        print("🤖 Model Response: ", generate_model_response(encoded_images[3], q4))
        
    print("=" * 60)
    print("🏁 MULTIMODAL EVALUATION COMPLETE")
