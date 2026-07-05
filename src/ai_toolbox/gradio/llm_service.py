"""
Service for interacting with the Llama Multimodal and Generation models using LiteLLM and LangChain.
"""

import logging
import sys
import os
import warnings

# 1. Cross-folder path patch to pull your central project config settings
flask_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'genai_flask_app'))
if flask_app_path not in sys.path:
    sys.path.append(flask_app_path)

from config import Config
from model import llm  # Your active master ChatLiteLLM instance
from langchain_core.messages import HumanMessage

# Set up logging tracking infrastructure
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress runtime configuration and layout warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
warnings.filterwarnings('ignore')


class LlamaVisionService:
    """
    Provides methods to interact with Llama vision-instruct generation models natively.
    """
    def __init__(self, model_id=None, project_id=None, region="us-south", 
                 temperature=0.2, top_p=0.6, api_key=None, max_tokens=2000):
        """
        Initialize the service and map parameters using LangChain's native execution bindings.
        """
        logger.info("Initializing modern LlamaVisionService.")
        
        # FIX: Check if we are running in local offline mode via our env config
        if Config.LLM_PROVIDER.lower() in ["ollama", "local"]:
            # Route strictly to an isolated ChatOllama instance, discarding cloud signature binds
            from langchain_ollama import ChatOllama
            self.vision_runner = ChatOllama(model=Config.MODEL_ID)
        else:
            # Route safely to LiteLLM for cloud tracks
            self.vision_runner = llm.bind(
                temperature=temperature,
                max_tokens=max_tokens
            )

    def generate_response(self, encoded_image, prompt):
        """
        Generate a response from the model based on an image and text prompt natively.
        """
        try:
            logger.info("Sending multimodal request to LLM with prompt length: %d", len(prompt))
            
            if not encoded_image:
                return "Error: Image payload missing or corrupt."
            
            # DONE: Create the unified multimodal content structure payload array
            message_content = [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
            
            # Wrap the payload inside a native LangChain HumanMessage object
            msg = HumanMessage(content=message_content)
            
            # DONE: Send the request to the model using .invoke()
            response = self.vision_runner.invoke([msg])
            
            # DONE: Extract and validate content response
            output_content = response.content.strip()
            
            # DONE: Check if response appears to be truncated (Hitting the max token ceiling)
            if response.response_metadata and response.response_metadata.get("finish_reason") == "length":
                logger.warning("🚨 Response was truncated due to max_tokens ceiling limit flags.")
                output_content += "\n\n[System Alert: This response was cut off due to generation token length constraints.]"
                
            return output_content
            
        except Exception as e:
            logger.error("Error generating response: %s", str(e))
            return f"Error generating response: {e}"
    
    def generate_fashion_response(self, user_image_base64, matched_row, all_items, 
                                 similarity_score, threshold=0.8):
        """
        Generate a fashion-specific response using role-based contextual prompt layouts.
        """
        try:
            # DONE: Generate a list of items with prices and links out of your dataset DataFrame matrix
            items_list = []
            
            # Loop through the items rows provided by the vector similarity search
            for _, row in all_items.iterrows():
                name = row.get("product_name", "Fashion Item")
                price = row.get("price", "N/A")
                link = row.get("click_url", "#")
                items_list.append(f"👗 {name} — Price: {price} | [View Details Plan Link]({link})")
            
            # DONE: Join items with clear visual separator newlines
            formatted_items_block = "\n".join(items_list)
            
            # DONE: Create role prompt based on similarity threshold strictness boundaries
            system_role = (
                "You are an expert personal fashion stylist assistant specializing in replicating celebrity outfits. "
                "Analyze the user's uploaded image and compare it directly to our verified matching item look catalog.\n\n"
            )
            
            if similarity_score >= threshold:
                context_prompt = (
                    f"Great news! We found a stellar direct match for this look (Confidence Match Score: {similarity_score*100:.1f}%).\n"
                    "Explain why this exact dataset product perfectly mirrors their look, comment on the style, "
                    "and introduce the individual items below with coordinator style advice."
                )
            else:
                context_prompt = (
                    f"We found an alternative match that shares a highly similar vibe (Confidence Match Score: {similarity_score*100:.1f}%).\n"
                    "Acknowledge that while it is not an identical match, it captures the same aesthetic theme. "
                    "Provide helpful suggestions on how to accessorize or complete the outfit using the pieces below."
                )
                
            complete_prompt = f"{system_role}{context_prompt}\n\nHere are the specific available matching catalog items:\n{formatted_items_block}\n\nStylist Response Layout:"
            
            # DONE: Send the compiled prompt payload to the vision execution model
            model_feedback = self.generate_response(user_image_base64, complete_prompt)
            
            # DONE: Ensure the structural items list string blocks are always appended as a fallback
            if "👗" not in model_feedback:
                logger.warning("⚠️ Model response omitted item links block. Appending catalog fallback strings manually.")
                model_feedback += f"\n\n### 🛒 Available Collection Catalog Items:\n{formatted_items_block}"
                
            return model_feedback
            
        except Exception as e:
            logger.error("Error generating fashion response: %s", str(e))
            return f"Error assembling stylist recommendations look panel: {e}"
