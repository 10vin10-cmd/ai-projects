import os
import sys
import warnings
import requests
import torch
from transformers import pipeline

# 1. Suppress framework and environment layer layout warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
warnings.filterwarnings('ignore')

# 2. Network Downloader Module
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hTqGqoC-LrW6S79HjuJUkg/trimmed-02.wav"
audio_file_path = "sample-meeting.wav"

if os.path.exists(audio_file_path):
    print(f"📦 '{audio_file_path}' already exists locally. Skipping download to prevent duplicates.")
else:
    print("⏳ File not found. Downloading audio asset from cloud repository...")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(audio_file_path, "wb") as file:
                file.write(response.content)
            print("✅ File downloaded successfully: 'sample-meeting.wav'")
        else:
            print(f"❌ Failed to download file. Status Code: {response.status_code}")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download file. Error: {e}")
        sys.exit(1)
        
# 3. Dynamic Hardware Acceleration Selection
# Checks if Apple Silicon GPU (MPS) is available, otherwise defaults to CPU
if torch.backends.mps.is_available():
    target_device = "mps"
    print("🚀 Hardware Accelerator Detected: Utilizing Apple Silicon GPU (MPS)")
else:
    target_device = "cpu"
    print("⚠️ Hardware Accelerator Missing: Defaulting to CPU processing")

import torch
from transformers import pipeline
import gradio as gr

# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
	    
	# Initialize Modern Whisper Speech-to-Text Pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
        device=target_device  # FIX: Routes computations to your Mac's GPU
    )
    print("🎙️ Transcribing audio file contents via Whisper Engine...")
    # Execute speech recognition using batches
    result = pipe(audio_file, batch_size=8)["text"]

    print("\n" + "="*60)
    # Clean up extra spacing flags and render transcription output text
    print("📝 TRANSCRIBED MEETING TEXT:")
    print("="*60)
    print(result.strip())
    print("="*60 + "\n")
    return result.strip()  # Clean up extra whitespace and return the transcription

# Set up Gradio interface
audio_input = gr.Audio(sources=["upload"], type="filepath", label="Upload Meeting Audio File")
output_text = gr.Textbox()  # Text output
# FIX: Ensure sources is passed as a list array string match


# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(fn=transcript_audio, 
					 inputs=audio_input, outputs=output_text, 
					 title="Audio Transcription App",
					 description="Upload the audio file")

# Launch the Gradio app
iface.launch(server_name="0.0.0.0", server_port=7860)


#transcript_audio(audio_file_path)  # Directly invoke the transcription function for testing