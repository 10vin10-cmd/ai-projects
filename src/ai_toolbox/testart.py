import os
import sys
import requests
from openai import OpenAI

# 1. Dynamically find and load your master .env configurations
flask_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'genai_flask_app'))
if flask_app_path not in sys.path:
    sys.path.append(flask_app_path)
from config import Config

# 2. Instantiate the official OpenAI Client
client = OpenAI()

def make_standalone_image():
    prompt_text = "a white siamese cat"
    print(f"🎨 Sending request to OpenAI using prompt: '{prompt_text}'...")
    
    try:
        # Requesting image generation
        response = client.images.generate(
            model="gpt-image-2",
            prompt=prompt_text,
            size="1024x1024",
            n=1
        )
        
        # FIX: Explicitly target array index position zero [0] to extract the URL string safely
        image_url = response.data[0].url
        print(f"🔗 Secure Cloud Link Received: {image_url}")
        
        # 3. Download the physical file to your local Mac hard drive
        print("📥 Downloading high-resolution image to local storage...")
        output_filename = "standalone_cat.png"
        image_bytes = requests.get(image_url).content
        
        with open(output_filename, "wb") as file:
            file.write(image_bytes)
            
        print(f"💾 Success! Image saved to your folder as '{output_filename}'")
        
        # 4. Force your MacBook Air to open the image using Preview app instantly
        os.system(f"open {output_filename}")
        
    except Exception as e:
        print(f"❌ API Execution Crash: {e}")

if __name__ == "__main__":
    make_standalone_image()
