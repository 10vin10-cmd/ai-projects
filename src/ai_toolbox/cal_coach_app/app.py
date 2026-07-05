"""
Modern Production Flask AI Server — Asynchronous Multi-Modal Nutrition Analyst Pipeline.
"""

import sys
import os
import base64
import time
import warnings
import concurrent.futures  
import markdown  
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

# Base Directory Anchors: Isolate the folder where app.py actively lives
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to 'src/ai_toolbox/'
ai_toolbox_dir = os.path.abspath(os.path.join(current_dir, '..'))
flask_config_path = os.path.abspath(os.path.join(ai_toolbox_dir, 'genai_flask_app'))
workspace_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

for path_route in [flask_config_path, ai_toolbox_dir, workspace_root]:
    if path_route not in sys.path:
        sys.path.insert(0, path_route)

warnings.filterwarnings('ignore')

from config import Config  
from model import llm
from langchain_core.messages import HumanMessage

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
app.secret_key = Config.SECRET_KEY

vision_runner = llm.bind(temperature=0.2, max_tokens=2000)

def generate_nutrition_assessment(encoded_image, user_query, assistant_prompt, file_extension=".jpg"):
    ext = file_extension.lower().strip('.')
    media_type = "png" if ext == "png" else "jpeg"
    base64_url = f"data:image/{media_type};base64,{encoded_image}"

    message_content = [
        {"type": "text", "text": f"{assistant_prompt.strip()}\n\nUser Question: {user_query}"},
        {"type": "image_url", "image_url": {"url": base64_url}}
    ]
    
    try:
        msg = HumanMessage(content=message_content)
        print("🤖 AI Vision Node: Dispatching multi-modal calorie assessment thread...")
        response = vision_runner.invoke([msg])
        processed_text = response.content.strip().replace("$", "\\$")
        return markdown.markdown(processed_text, extensions=['extra', 'sane_lists'])
    except Exception as e:
        print(f"❌ Vision Pipeline Exception: {e}")
        return f"<p class='error-msg'>An error occurred while generating the nutrition assessment: {e}</p>"

@app.route("/", methods=["GET"])
def index():
    """Renders the primary web dashboard interface instantly with zero wait times."""
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Isolated API endpoint called asynchronously via Javascript fetch.
    Bypasses standard page-refresh browser timeout drops entirely!
    """
    user_query = request.form.get("user_query", "Analyze this meal.")
    uploaded_file = request.files.get("file")

    if not uploaded_file or uploaded_file.filename == '':
        return jsonify({"success": False, "error": "Please select a meal photo before submitting."}), 400

    try:
        # Encode image bytes stream in-memory
        bytes_data = uploaded_file.read()
        encoded_image = base64.b64encode(bytes_data).decode("utf-8")
        _, file_ext = os.path.splitext(uploaded_file.filename)

        assistant_prompt = """You are an expert nutritionist. Your task is to analyze the food items displayed in the image and provide a detailed nutritional assessment using the following format:
        1. **Identification**: List each identified food item clearly.
        2. **Portion Size & Calorie Estimation**: For each item specify the portion size and calorie estimation.
        3. **Total Calories**: Total calories summary.
        4. **Nutrient Breakdown**: Protein, Carbohydrates, Fats, Vitamins, Minerals.
        5. **Health Evaluation**: Meal healthiness review paragraph.
        6. **Disclaimer**: Include the general approximate values disclaimer text.
        Format response cleanly exactly matching template spacing boundaries."""

        # Execute heavy vision calculations in a background thread task
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(generate_nutrition_assessment, encoded_image, user_query, assistant_prompt, file_ext)
            while not future.done():
                time.sleep(0.05)
            html_report = future.result()

        # Return the processed HTML payload structure inside a safe, structured JSON packet
        return jsonify({"success": True, "html_output": html_report, "user_query": user_query})

    except Exception as e:
        print(f"❌ API Exception context block: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Production Async Cal Coach Server on http://127.0.0.1:5001")
    app.run(host='127.0.0.1', port=5001, debug=True)
