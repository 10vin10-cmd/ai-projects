import os
import sys
import httpx
import requests 
import time  
import concurrent.futures
import gradio as gr
from PIL import Image
from openai import OpenAI

# Dynamic absolute path mapping to load master .env files safely
current_dir = os.path.dirname(os.path.abspath(__file__))
flask_app_path = os.path.abspath(os.path.join(current_dir, '..', 'genai_flask_app'))
if flask_app_path not in sys.path: 
    sys.path.append(flask_app_path)
from config import Config

# Initialize client with an explicit network timeout budget
client = OpenAI(
    timeout=httpx.Timeout(120.0, read=90.0, write=20.0, connect=20.0)
)

def call_openai_edit_api(input_img_path, prompt_string):
    """
    Background worker task that targets OpenAI's image transformation
    pipeline without blocking Gradio's web engine context thread.
    """
    try:
        # 1. Open and optimize the user's actor image natively.
        # OpenAI requires editing reference images to be square PNG files under 4MB.
        with Image.open(input_img_path) as img:
            optimized_img = img.convert("RGBA").resize((1024, 1024))
            temp_png_path = "optimized_actor_input.png"
            optimized_img.save(temp_png_path, "PNG")

        # 2. Dispatch payload to OpenAI's image modification cluster
        # Using the standard image endpoint with file handles transforms the actor natively
        with open(temp_png_path, "rb") as image_file:
            response = client.images.create_variation(
                image=image_file,
                model="gpt-image-2", # Uses your premium active image generation model family
                n=1,
                size="1024x1024"
            )
        
        # Cleanup temporary optimized file asset immediately
        if os.path.exists(temp_png_path):
            os.remove(temp_png_path)
            
        return response
    except Exception as e:
        return f"OPENAI_API_CRASH: {str(e)}"

def modify_actor_image(actor_image, structural_prompt):
    """
    Orchestrates the asynchronous generation lifecycle, tracking live second counter 
    heartbeats to keep the Gradio socket alive during the 35-second cloud compute window.
    """
    if actor_image is None:
        raise gr.Error("Please upload an actor image reference file first!")
    if not structural_prompt:
        raise gr.Error("Please provide text instructions explaining how to modify the actor!")
        
    try:
        print(f"🎬 Initializing Actor Modification for prompt: '{structural_prompt}'")
        yield None, "⏳ Optimizing actor image dimensions and alpha channels..."
        
        # Initialize thread worker pool tasks
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(call_openai_edit_api, actor_image, structural_prompt)
            
            # Keep-Alive Heartbeat Counter Loop
            counter = 0
            while not future.done():
                counter += 1
                progress_text = f"🧬 Swapping style layers: Rendering modified actor scene... ({counter}s passed)"
                yield None, progress_text
                time.sleep(1.0)
            
            response = future.result()

        # Capture and map account level credit or cluster barriers
        if isinstance(response, str) and "OPENAI_API_CRASH" in response:
            raise gr.Error(f"OpenAI Account Barrier: {response.replace('OPENAI_API_CRASH: ', '')}")

        response_dict = response.model_dump()

        if response_dict and "data" in response_dict and len(response_dict["data"]) > 0:
            image_url = response_dict["data"][0]["url"]
            print(f"🔗 Secure link retrieved from variation cluster: {image_url}")
            
            yield None, "📥 Downloading high-fidelity character generation to your Mac..."
            
            # Break browser caching traps using distinct timestamps
            timestamp = int(time.time())
            local_filename = f"actor_render_{timestamp}.png"
            image_bytes = requests.get(image_url).content
            
            with open(local_filename, "wb") as file:
                file.write(image_bytes)
                
            print(f"💾 Rendered scene saved successfully to disk as: {local_filename}")
            yield local_filename, "✨ Generation Complete! Your custom character look has loaded below."
        else:
            raise gr.Error("OpenAI processed the request but returned an invalid payload structure.")
            
    except Exception as e:
        print(f"❌ Core API Error Logged: {str(e)}")
        raise gr.Error(f"API Execution Failure: {str(e)}")

# ══════════════════════════════════════════════════════════════════
# GRADIO INTERFACE LAYOUT — MULTI-MODAL CANVAS
# ══════════════════════════════════════════════════════════════════
with gr.Blocks(title="AI Character & Actor Style Modifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 AI Character & Actor Style Modifier")
    gr.Markdown("Upload an image of an actor or subject, pass style text modifications, and render them inside completely new scenes!")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Actor Image File (Reference Source)", type="filepath")
            style_prompt = gr.Textbox(
                label="Style Modification Prompt Instructions:", 
                placeholder="e.g., Change costume to a futuristic neon cyberpunk spacesuit, photorealistic cinematic lighting"
            )
            generate_btn = gr.Button("Transform Character Scene", variant="primary")
            status_banner = gr.Markdown("Ready")
            
        with gr.Column(scale=2):
            output_image = gr.Image(label="Rendered AI Consistent Character Canvas", type="filepath")

    # Wire button listener outputs matching function yield sequencing
    generate_btn.click(
        fn=modify_actor_image, 
        inputs=[input_image, style_prompt], 
        outputs=[output_image, status_banner]
    )

if __name__ == "__main__":
    print("🚀 Launching Multi-Modal Image-to-Image Modifier portal on local loopback...")
    demo.launch(server_name="127.0.0.1", server_port=7861)
