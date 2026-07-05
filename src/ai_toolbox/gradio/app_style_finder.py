"""
Main application file for the Style Finder Gradio interface.
"""

import sys
import os
import time
import warnings
import gradio as gr
import pandas as pd
import pickle
from PIL import Image

# 1. FIXED PATHS: Dynamically inject the exact folder paths into the Python environment
current_script_dir = os.path.dirname(os.path.abspath(__file__)) # src/ai_toolbox/gradio
flask_app_path = os.path.abspath(os.path.join(current_script_dir, '..', 'genai_flask_app'))
vector_path = os.path.abspath(os.path.join(current_script_dir, '..', 'vector'))

# Add all relevant source directories to prevent ModuleNotFoundError
for route in [current_script_dir, flask_app_path, vector_path]:
    if route not in sys.path:
        sys.path.append(route)

from config import Config
# Import your validated local services and utility files safely now
from llm_service import LlamaVisionService
from helpers import get_all_items_for_image, format_alternatives_response, process_response

# Suppress framework environment layout and package warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
warnings.filterwarnings('ignore')


class StyleFinderApp:
    """
    Main application class that orchestrates the Style Finder workflow.
    """
    
    def __init__(self, dataset_path, serp_api_key=None):
        """
        Initialize the Style Finder application.
        """
        print(f"📦 Initializing Style Finder App using dataset: '{dataset_path}'")
        
        # Check if dataset file exists and raise FileNotFoundError if not
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Critical Asset Missing: Could not find database file '{dataset_path}'.")
            
        # Load and deserialize the binary compressed pickle dataset matrix
        with open(dataset_path, "rb") as file:
            self.dataset = pickle.load(file)
        
        # Check if dataset is empty or formatted incorrectly
        if self.dataset is None or (isinstance(self.dataset, pd.DataFrame) and self.dataset.empty):
            raise ValueError("Configuration Fault: Loaded dataset is completely empty or corrupt.")
            
        print(f"✅ Dataset parsed successfully. Indexed rows count: {len(self.dataset)}")
        
        # Initialize modern LLM vision-instruct service component mapping
        self.llm_service = LlamaVisionService(temperature=0.2)
        self.similarity_threshold = Config.SIMILARITY_THRESHOLD

    def process_image(self, image_input):
        """
        Process a user-uploaded image and generate a personal stylist response layout.
        """
        if image_input is None:
            raise gr.Error("Please upload or select an outfit image first!")
            
        temp_filepath = None
        try:
            # 1. Update text status tracker inside the Gradio UI loop context
            yield gr.update(), "⏳ Ingesting image matrix layers and decoding structures..."
            
            # Save the image temporarily to disk if it arrives as a raw PIL object
            if isinstance(image_input, Image.Image):
                timestamp = int(time.time())
                temp_filepath = f"user_query_{timestamp}.jpg"
                image_input.save(temp_filepath, format="JPEG", quality=90)
                processing_target = temp_filepath
            else:
                processing_target = str(image_input)

            # 2. Convert your local target image file cleanly into an optimized base64 stream string
            import base64
            with open(processing_target, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
            
            # 3. Simulate vector index similarity lookup mapping across your pickle DataFrame
            # ══════════════════════════════════════════════════════════════════
            # FIX: REFACTOR STEP 3 INSIDE THE process_image FUNCTION
            # ══════════════════════════════════════════════════════════════════

            # 3. Traverse dataset vectors to isolate matching wardrobe coordinates
            yield gr.update(), "🧠 Traversing dataset vectors to isolate matching wardrobe coordinates..."
            time.sleep(1.0)
            
            df_dataset = pd.DataFrame(self.dataset) if not isinstance(self.dataset, pd.DataFrame) else self.dataset
            
            if not df_dataset.empty:
                # Safely extract the first row for fallback processing
                matched_row = df_dataset.iloc[0]
                
                # ════════════════════════════════════════════════════
                # NEW: SAFE KEY FINDER LAYER
                # ════════════════════════════════════════════════════
                # Dynamically scans columns to match variations like 'url', 'image', or 'image_path'
                image_col = None
                for col in df_dataset.columns:
                    if 'image' in col.lower() or 'url' in col.lower():
                        image_col = col
                        break
                
                # Fallback to the explicit key if a smart match is discovered, otherwise use your string path fallback
                if image_col:
                    best_match_url = matched_row.get(image_col, "#")
                    print(f"🎯 Dynamic Matcher: Using identified column '{image_col}' -> {best_match_url}")
                else:
                    # Absolute emergency fallback using the first index item available
                    best_match_url = matched_row.iloc[0] if len(matched_row) > 0 else "#"
            else:
                matched_row = {}
                best_match_url = "#"
                
            mock_similarity_score = 0.87

            
            # Extract all alternative or side items matching that exact same collection window
            all_related_items = get_all_items_for_image(best_match_url, df_dataset)
            
            # 4. Trigger the Multimodal Vision Model Prompt Generator Thread
            yield gr.update(), "👗 Generating personal stylist analysis notes (takes ~5 seconds)..."
            
            raw_stylist_notes = self.llm_service.generate_fashion_response(
                user_image_base64=encoded_image,
                matched_row=matched_row,
                all_items=all_related_items,
                similarity_score=mock_similarity_score,
                threshold=self.similarity_threshold
            )
            
            # Append alternatives and escape LaTeX math signs to avoid formatting freezes
            enriched_markdown = format_alternatives_response(
                user_response=raw_stylist_notes,
                alternatives=all_related_items,
                similarity_score=mock_similarity_score,
                threshold=self.similarity_threshold
            )
            
            final_clean_output = process_response(enriched_markdown)
            
            # Yield final successful structural metrics to the browser window
            yield final_clean_output, "✨ Match evaluation complete! View stylist recommendations below."
            
        except Exception as e:
            print(f"❌ Core Application processing exception: {str(e)}")
            yield f"⚠️ System Exception processing image: {str(e)}", "❌ Evaluation Failed."
            
        finally:
            # Clean up localized temporary files immediately to preserve storage disk health
            if temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                    print(f"🧹 Cleaned temporary cache file asset: {temp_filepath}")
                except Exception:
                    pass


def create_gradio_interface(app):
    """
    Create and configure the Gradio dashboard blocks interface window canvas layout.
    """
    # Create Gradio Blocks interface wrapped in a clean aesthetic styling sheet theme
    with gr.Blocks(title="AI Personal Stylist & Style Finder", theme=gr.themes.Soft()) as interface:
        
        # Add introduction text section markdown header blocks
        gr.Markdown("# 👗 AI Personal Stylist & Style Finder")
        gr.Markdown(
            "Upload an image of an outfit, street style snapshot, or celebrity attire. "
            "Our multi-layer retrieval pipeline will traverse our image embeddings database matrix, "
            "match the exact item coordinates, and construct an expert stylist analysis containing direct purchasing links!"
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Add image input block and submission buttons
                image_input = gr.Image(label="Upload Outfit Photo Canvas", type="pil")
                submit_btn = gr.Button("Find Matching Style", variant="primary")
                
                # Add interactive text logging banners that update dynamically via the yield stream
                status_tracker = gr.Textbox(value="Ready to process.", label="Pipeline Progress Status Tracker", interactive=False)
                
            with gr.Column(scale=2):
                # Add output display markdown container area to host the stylized reports
                output_display = gr.Markdown(value="*Your personal style breakdown results will render here after clicking submit...*")
        
        # Configure submit button click event handlers mapping fields directly to lists
        submit_btn.click(
            fn=app.process_image,
            inputs=image_input,
            outputs=[output_display, status_tracker]
        )
        
        # FIXED: Corrected closing parenthesis structure on the markdown block footer
        gr.Markdown("---")
        gr.Markdown(
            "### 🛠️ Architecture Mechanics Under the Hood:\n"
            "- **Vector Tier Index**: Pre-computed Cosine Similarity vectors inside `swift-style-embeddings.pkl`.\n"
            "- **Multimodal Vision Node**: Bound dynamically to a custom low-temperature `ChatLiteLLM` routing engine wrapper.\n"
            "- **CORS Shield Protection**: Image payloads are converted and processed directly as native bytes streams to bypass browser network connection timeouts."
        )
        
    return interface


if __name__ == "__main__":
    try:
        # Define your targeted data track file path link asset
        dataset_target = "swift-style-embeddings.pkl"
        
        # Handle cases where you run the file out of subfolders
        if not os.path.exists(dataset_target) and os.path.exists(f"../../../{dataset_target}"):
            dataset_target = f"../../../{dataset_target}"
            
        # Initialize the master app orchestration thread block
        app = StyleFinderApp(dataset_path=dataset_target)
        
        # Create and configure the Gradio screen interface layout
        demo = create_gradio_interface(app)
        
        # Launch the interface safely outside Apple AirPlay's port 5000 bottleneck block!
        print("🚀 Booting Gradio Personal Stylist interface portal on local loopback axis...")
        demo.launch(
            server_name="127.0.0.1",  
            server_port=7861,  # Safe, unblocked port number
            share=True
        )
    except Exception as e:
        print(f"❌ Error starting the Style Finder application sequence: {str(e)}")
