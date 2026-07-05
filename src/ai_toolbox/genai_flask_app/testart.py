import os
import sys
import requests
import json
from openai import OpenAI

# 1. Dynamically locate and load your master .env configurations
flask_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'genai_flask_app'))
if flask_app_path not in sys.path:
    sys.path.append(flask_app_path)
from config import Config # Automatically handles your OPENAI_API_KEY initialization

# 2. Instantiate the official OpenAI Client
client = OpenAI()

def make_standalone_image():
    prompt_text = "a white siamese cat"
    print(f"🎨 Dispatching generation task for prompt: '{prompt_text}'")
    
    try:
        # Requesting image generation via the premium gpt-image-2 model family
        response_obj = client.images.generate(
            model="gpt-image-2",
            prompt=prompt_text,
            size="1024x1024",
            n=1
        )
        
        # --- THE DIAGNOSTIC LAYER ---
        # Convert response into a raw string dictionary format to inspect its anatomy
        raw_dict = response_obj.model_dump()
        print("\n🔍 --- DEBUG: RAW OPENAI RESPONSE STRUCTURE ---")
        print(json.dumps(raw_dict, indent=2))
        print("───────────────────────────────────────────────\n")
        
        image_url = None
        image_bytes = None
        
        # FALLBACK 1: Try dictionary parsing with position index 0 [0]
        if "data" in raw_dict and len(raw_dict["data"]) > 0:
            first_item = raw_dict["data"][0]
            
            # Check for URL string link
            if "url" in first_item and first_item["url"]:
                image_url = first_item["url"]
                print(f"🔗 Match Found: Extracted Web URL -> {image_url}")
            
            # Check for raw Base64 JSON text data stream
            elif "b64_json" in first_item and first_item["b64_json"]:
                print("📦 Match Found: Extracted Base64 Image String.")
                import base64
                image_bytes = base64.b64decode(first_item["b64_json"])
                
        # FALLBACK 2: Try object dot-notation parsing as a secondary rescue
        if not image_url and not image_bytes:
            if hasattr(response_obj, 'data') and len(response_obj.data) > 0:
                if hasattr(response_obj.data[0], 'url') and response_obj.data[0].url:
                    image_url = response_obj.data[0].url
                    print(f"🔗 Object Fallback Match: Extracted Web URL -> {image_url}")
                elif hasattr(response_obj.data[0], 'b64_json') and response_obj.data[0].b64_json:
                    import base64
                    print("📦 Object Fallback Match: Extracted Base64 String.")
                    image_bytes = base64.b64decode(response_obj.data[0].b64_json)

        # 3. Process the file payload creation based on which fallback succeeded
        output_filename = "standalone_cat.png"
        
        if image_url:
            print("📥 Downloading high-resolution image to local storage...")
            image_bytes = requests.get(image_url).content
            
        if image_bytes:
            with open(output_filename, "wb") as file:
                file.write(image_bytes)
            print(f"💾 Success! Image saved to your folder as '{output_filename}'")
            
            # Force your MacBook Air to open the image using the Preview app instantly
            os.system(f"open {output_filename}")
        else:
            print("❌ Failure: Every fallback parser evaluated to None.")
            print("Please inspect the raw JSON structure printed above to find the missing key identifier!")
            
    except Exception as e:
        print(f"❌ Core API Error Logged: {str(e)}")

if __name__ == "__main__":
    make_standalone_image()
