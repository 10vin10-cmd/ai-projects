from gtts import gTTS
from IPython.display import Audio
import io
import os
import sys

# Step up one directory layer out of 'vector', then step into 'genai_flask_app'
flask_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'genai_flask_app'))

# Append this folder location dynamically to Python's file routing engine
if flask_app_path not in sys.path:
    sys.path.append(flask_app_path)

# 1. Import your active central configuration entities
from model import llm  # Using your active master ChatLiteLLM instance
from config import Config  

# Function to generate an educational story using the dynamic LiteLLM engine
def generate_story(topic):
    # Construct a detailed prompt
    prompt = f"""Write an engaging and educational story about {topic} for beginners. 
    Use simple and clear language to explain basic concepts. 
    Include interesting facts and keep it friendly and encouraging. 
    The story should be around 200-300 words and end with a brief summary of what we learned. 
    Make it perfect for someone just starting to learn about this topic."""
    
    # FIX: Swapped out the deprecated .generate_text() for the modern .invoke() 
    # and extracted the raw plain text string using .content
    response = llm.invoke(prompt)
    return response.content

# ----------------------------------------------------
# EXECUTION WORKFLOW
# ----------------------------------------------------
topic = "An interesting short story of Shivaji to tell my 12 year old son"#"the life cycle of butterflies"

print(f"⏳ Generating educational story about: '{topic}'...")
story = generate_story(topic)

print("\n📖 GENERATED STORY:\n", story)
print("-" * 60)

# Initialize text-to-speech with the generated story text string
print("🎙️ Synthesizing text to speech audio...")
tts = gTTS(story, lang='en', slow=False)

# Save the audio to a bytes buffer securely in memory
audio_bytes = io.BytesIO()
tts.write_to_fp(audio_bytes)
audio_bytes.seek(0)

# FIX: Used getvalue() to ensure the complete audio byte sequence is passed cleanly
# to the IPython notebook player widget
Audio(audio_bytes.getvalue(), autoplay=False)  # Set autoplay to False to avoid auto-playing in notebooks

# Save a physical audio file directly into your project folder
#output_file = "butterfly_story.mp3"
#tts.save(output_file)

#print(f"🔊 Audio file saved successfully as '{output_file}'")

# OPTIONAL MAC AUTO-PLAY: Automatically forces your Mac speaker to play it immediately!

#os.system(f"open {output_file}")