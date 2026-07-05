"""
Utility functions for the Style Finder application.
"""

import logging
import re
import pandas as pd

# Set up logging tracking infrastructure
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_items_for_image(image_url, dataset):
    """
    Get all items related to a specific image from the dataset DataFrame.
    """
    if not isinstance(dataset, pd.DataFrame):
        logger.error("Dataset provided is not a valid pandas DataFrame matrix.")
        return pd.DataFrame()

    # ════════════════════════════════════════════════════
    # NEW: DYNAMIC UTILITY COLUMN MATCH LAYER
    # ════════════════════════════════════════════════════
    # Checks for capitalization differences (like 'Image URL') in the column headers
    target_column = None
    for col in dataset.columns:
        if 'image' in col.lower() or 'url' in col.lower():
            target_column = col
            break

    # If the column finder fails, default back to 'Image URL' as the verified match
    if not target_column:
        target_column = 'Image URL'

    print(f"📊 Utility Matcher: Filtering items collection by key '{target_column}'")

    # Filter the dataset using the dynamically identified column name securely
    matched_items = dataset[dataset[target_column].astype(str).str.strip() == str(image_url).strip()]
    
    logger.info("Found %d distinct clothing items linked to image path URL.", len(matched_items))
    return matched_items


def format_alternatives_response(user_response, alternatives, similarity_score, threshold=0.8):
    """
    Append alternatives and styling choices to the user response in a structured layout.
    
    Args:
        user_response (str): Original conversational markdown response from the model
        alternatives (dict or DataFrame): Collection of alternatives for each outfit piece
        similarity_score (float): Similarity score (0.0 to 1.0) of the vector match
        threshold (float): Limit cutoff for determining exact match quality (default: 0.8)
        
    Returns:
        str: Enhanced markdown response with alternative items appended
    """
    # DONE: Check if user_response is problematic and build a foundational backup string if needed
    if not user_response or len(user_response.strip()) < 10:
        logger.warning("Base response missing or corrupt. Injecting standard fallback header.")
        base_output = "# Personal Stylist Evaluation\n\nHere are the catalog components curated for your look:"
    else:
        base_output = user_response.strip()
        
    # DONE: Add dynamic section headers based on similarity score match thresholds
    enhanced_parts = [base_output, ""]
    
    if similarity_score >= threshold:
        enhanced_parts.append("### 🌟 Curated Collection Components")
        enhanced_parts.append("We found a high-confidence style match! Here are the exact pieces available in our database to replicate this exact look:")
    else:
        enhanced_parts.append("### 🔄 Alternative Wardrobe Recommendations")
        enhanced_parts.append("While not an exact match, these alternative selection pieces capture the identical aesthetic energy and silhouette:")
        
    enhanced_parts.append("")
    
    # DONE: Parse, count, and format individual alternative row elements
    item_counter = 0
    max_alternatives_limit = 5  # Enforces a clean screen UI boundary
    
    # Handle item parsing if alternatives are passed as a Pandas DataFrame matrix
    if isinstance(alternatives, pd.DataFrame):
        for _, row in alternatives.iterrows():
            if item_counter >= max_alternatives_limit:
                break
                
            name = row.get("product_name", "Fashion Accessory")
            price = row.get("price", "N/A")
            link = row.get("click_url", "#")
            brand = row.get("brand", "Exclusive Designer")
            
            # Escape currency tokens during extraction to protect Markdown matrices
            if isinstance(price, str):
                price = price.replace("$", "\\$")
                
            # DONE: Format each alternative item with clean Markdown syntax anchors
            item_markdown = f"- **{name}** by *{brand}*\n  - Price: {price}\n  - 🔗 [Secure Product Page Link]({link})"
            enhanced_parts.append(item_markdown)
            item_counter += 1
            
    # Handle dictionary array formats safely as a fallback configuration
    elif isinstance(alternatives, list):
        for item in alternatives:
            if item_counter >= max_alternatives_limit:
                break
            if isinstance(item, dict):
                name = item.get("product_name", "Fashion Piece")
                price = str(item.get("price", "N/A")).replace("$", "\\$")
                link = item.get("click_url", "#")
                enhanced_parts.append(f"- **{name}** — {price} | [Product Link]({link})")
                item_counter += 1

    if item_counter == 0:
        enhanced_parts.append("_No additional matching coordinates are currently active in this collection catalog vector slot._")
        
    # DONE: Return the fully enhanced markdown layout string
    return "\n".join(enhanced_parts)

def process_response(response: str) -> str:
    """
    Process and escape problematic formatting characters inside the generated response text.
    
    Args:
        response (str): The raw original response text string directly out of the model
        
    Returns:
        str: Processed response with escaped symbols and standardized header markdown layout
    """
    if not response:
        logger.warning("Empty response received")
        return "# Fashion Analysis\n\nNo detailed analysis was generated. Please refer to the item details below."
    
    # Check for alignment boundary rejection messages
    rejection_phrases = [
        "I'm not able to provide",
        "I cannot provide",
        "I apologize, but I cannot",
        "I don't feel comfortable",
        "violated our content policy"
    ]
    
    # If the model rejected the vision content but item arrays are present, extract them safely
    if any(phrase in response for phrase in rejection_phrases):
        logger.warning("Model safety cutoff activated, extracting structured item data fallbacks")
        
        items_section = None
        if "ITEM DETAILS:" in response:
            items_section = "## Item Details\n\n" + response.split("ITEM DETAILS:")[1].strip()
        elif "SIMILAR ITEMS:" in response:
            items_section = "## Similar Items\n\n" + response.split("SIMILAR ITEMS:")[1].strip()
        
        if items_section:
            # Reformat item bullets with clean standard Markdown markers
            formatted_items = re.sub(r'^\* ', '- ', items_section, flags=re.MULTILINE)
            return "# Fashion Analysis\n\nHere are the available items matched from your uploaded layout image:\n\n" + formatted_items
        else:
            return "# Fashion Analysis\n\nContent query restricted by provider. Review available matching catalog links below."
    
    # Escape pricing dollar signs ($) to prevent LaTeX equation rendering bugs inside Gradio/Streamlit UI views
    processed = response.replace("$", "\\$")
    
    # Ensure raw legacy tracking block strings are swapped into beautiful H2 Section Headers
    if "ITEM DETAILS:" in processed:
        processed = processed.replace("ITEM DETAILS:", "## Item Details")
    
    if "SIMILAR ITEMS:" in processed:
        processed = processed.replace("SIMILAR ITEMS:", "## Similar Items")
    
    # Enforce an overarching clean Title Header layout if the model omitted it
    if not processed.strip().startswith("#"):
        processed = "# Fashion Analysis\n\n" + processed.strip()
    
    # Ensure all list bullets are mapped to standard unified dash markers ('- ')
    processed = re.sub(r'^\* ', '- ', processed, flags=re.MULTILINE)
    
    return processed
