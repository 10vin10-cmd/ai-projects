import os
import sys
import httpx
import requests 
import time  
import concurrent.futures
import gradio as gr
from openai import OpenAI
import base64

# Dynamic absolute path mapping to load master .env files safely
flask_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'genai_flask_app'))
if flask_app_path not in sys.path: 
    sys.path.append(flask_app_path)
from config import Config

# Initialize your client safely with a generous timeout budget
client = OpenAI(
    timeout=httpx.Timeout(120.0, read=90.0, write=20.0, connect=20.0)
)

def call_openai_api(prompt_string):
    """
    Isolated worker task wrapped in a strict try/except layout to ensure
    any backend API, model, or billing error is surfaced cleanly to the main thread.
    """
    try:
        response = client.images.generate(
            model="gpt-image-2", 
            prompt=prompt_string,
            size="1024x1024",
            n=1
        )
        return response
    except Exception as e:
        return f"OPENAI_API_CRASH: {str(e)}"

import base64  # Ensure this is imported at the very top of your script file!

def create_dalle_image(user_prompt):
    if not user_prompt:
        raise gr.Error("Please type a descriptive prompt first!")
        
    try:
        print(f"🎨 Dispatching generation task for prompt: '{user_prompt}'")
        
        # 1. Clear out any old images and update the text status immediately
        yield None, "⏳ Connecting to OpenAI generation cluster..."
        
        # 2. Initialize our secure thread pool background worker
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(call_openai_api, user_prompt)
            
            # 3. Dynamic Live Counter Loop (Sends heartbeats to prevent web timeouts)
            counter = 0
            while not future.done():
                counter += 1
                progress_text = f"🎨 Processing pixels: Working on image layers... ({counter}s passed)"
                yield None, progress_text
                time.sleep(1.0)
            
            # Retrieve the payload from our completed background worker thread
            response = future.result()

        # 4. Check if the thread worker returned an error string instead of an object
        if isinstance(response, str) and "OPENAI_API_CRASH" in response:
            raise gr.Error(f"OpenAI Account Barrier: {response.replace('OPENAI_API_CRASH: ', '')}")

        # 5. Convert response object layout into a standard Python dictionary layout
        if not response or not hasattr(response, 'model_dump'):
            raise gr.Error(f"Unexpected response format from background worker: {type(response)}")
            
        response_dict = response.model_dump()

        # 6. FIX: Extract the payload checking for BOTH URL and Base64 structures
        if response_dict and "data" in response_dict and len(response_dict["data"]) > 0:
            first_image_data = response_dict["data"][0]
            
            # Create a unique filename using a timestamp to smash Gradio's image cache!
            timestamp = int(time.time())
            local_filename = f"art_{timestamp}.png"
            
            # CASE A: Check if OpenAI returned a standard Web URL link
            if "url" in first_image_data and first_image_data["url"]:
                image_url = first_image_data["url"]
                print(f"🔗 Received cloud link asset route: {image_url}")
                yield None, "📥 Downloading high-resolution image to your local Mac storage..."
                image_bytes = requests.get(image_url).content
                
            # CASE B: Check if OpenAI returned a Base64 encoded string format instead
            elif "b64_json" in first_image_data and first_image_data["b64_json"]:
                print("📦 Received raw Base64 data streaming asset route.")
                yield None, "💾 Decoding and extracting image bytes natively in memory..."
                # Decode the text matrix directly into raw binary image bytes
                image_bytes = base64.b64decode(first_image_data["b64_json"])
                
            else:
                raise gr.Error("OpenAI processed the request but returned no readable image link or base64 keys.")
                
            # 7. Write the binary image bytes directly to your physical local file
            with open(local_filename, "wb") as file:
                file.write(image_bytes)
                
            print(f"💾 Image saved successfully to disk as: {local_filename}")
            
            # 8. Yield the uniquely named local file path string directly
            yield local_filename, "✨ Generation Complete! Your artwork has been loaded below."
        else:
            raise gr.Error("OpenAI processed the request but returned an empty payload structure.")
            
    except Exception as e:
        print(f"❌ Core API Error Logged: {str(e)}")
        raise gr.Error(f"API Execution Failure: {str(e)}")


# ══════════════════════════════════════════════════════════════════
# GRADIO INTERFACE LAYOUT
# ══════════════════════════════════════════════════════════════════
with gr.Blocks(title="AI Image Generator") as demo:
    gr.Markdown("# 🎨 Modern Image Creator Pipeline")
    
    with gr.Row():
        input_text = gr.Textbox(
            label="Describe your target image...", 
            placeholder="e.g., a white siamese cat playing with blue yarn"
        )
        generate_btn = gr.Button("Generate Art", variant="primary")
        
    status_banner = gr.Markdown("Ready")
    output_image = gr.Image(label="Generated AI Art", type="filepath")

    generate_btn.click(
        fn=create_dalle_image, 
        inputs=input_text, 
        outputs=[output_image, status_banner]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
