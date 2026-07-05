"""
Multi-Modal Voice-to-Voice Audio Translation and Synthesis Pipeline.
"""

import sys
import os
import time
import warnings
import torch
import gradio as gr
from transformers import pipeline
from gtts import gTTS

# 1. Path patch configuration to map your centralized model configurations
current_dir = os.path.dirname(os.path.abspath(__file__))
flask_app_path = os.path.abspath(os.path.join(current_dir, '..', 'genai_flask_app'))
if flask_app_path not in sys.path:
    sys.path.append(flask_app_path)

from config import Config
from model import llm  # Your active central ChatLiteLLM instance
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Suppress system layout and environment package warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════
# FIX: REFACTOR THIS FUNCTION INSIDE THE voice_translator.py FILE
# ══════════════════════════════════════════════════════════════════

def translate_text_engine(source_text, target_language):
    """
    Invokes your active LLM via LiteLLM to translate the raw transcript text string
    into the designated target language structure.
    """
    system_prompt = (
        "You are an expert real-time human translator. Your task is to translate the user's "
        "spoken transcript text into the requested target language accurately.\n"
        "RULES:\n"
        "- Maintain the original emotional tone and context rules.\n"
        "- Output ONLY the final translated text string. Do NOT add conversational pleasantries, "
        "markdown headers, or notes like 'Here is your translation:'."
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Translate the following text into {language}:\n\n{text}")
    ])
    
    # ✅ FIX: Use .bind() directly on the llm object inside the sequence pipeline compilation.
    # This securely isolates the 0.1 low-temperature parameter node!
    translation_chain = prompt_template | llm.bind(temperature=0.1) | StrOutputParser()
    
    # Invoke the cleanly bound sequence natively
    translated_output = translation_chain.invoke({
        "language": target_language,
        "text": source_text
    })
    return translated_output.strip()

# ══════════════════════════════════════════════════════════════════
# FIX: ALIGN ALL GENERATOR YIELDS IN THE voice_translator.py FILE
# ══════════════════════════════════════════════════════════════════

def run_voice_translation_pipeline(audio_filepath, target_lang_selection):
    if not audio_filepath:
        raise gr.Error("Please upload or record an audio file first!")
        
    try:
        # PHASE 1: LOCAL AUDIO TRANSCRIPTION
        # FIX: Added a 4th 'None' to keep the audio player element unblocked
        yield "🎙️ Processing audio: Booting local Whisper Transcription Engine...", None, None, None
        
        target_device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model="distil-whisper/distil-medium.en",
            chunk_length_s=30,
            device=target_device
        )
        
        print("🎙️ Whisper pipeline reading input bytes...")
        raw_transcription = pipe(audio_filepath, batch_size=8)["text"].strip()
        print(f"📝 Raw Source Transcript captured: {raw_transcription}")
        
        if not raw_transcription:
            raise gr.Error("Whisper completed processing but detected absolute silence.")

        # PHASE 2: CONTEXTUAL LLM TEXT TRANSLATION
        # FIX: Added a 4th 'None' value to balance the stream output arguments
        yield f"🧠 Text isolated. Sending transcript to {Config.LLM_PROVIDER.upper()} for Translation...", raw_transcription, None, None
        
        translated_text = translate_text_engine(raw_transcription, target_lang_selection)
        print(f"🌐 Final Translated text generated: {translated_text}")

        # PHASE 3: TEXT-TO-SPEECH SYNTHESIS ENGINE
        # FIX: Added a 4th 'None' value to maintain the strict 4-tuple balance requirement
        yield f"🎙️ Synthesis Loop: Rendering translated text into a voice file...", raw_transcription, translated_text, None
        
        lang_map = {
            "Spanish": {"code": "es", "tld": "com"},
            "French": {"code": "fr", "tld": "com"},
            "Hindi": {"code": "hi", "tld": "co.in"},
            "German": {"code": "de", "tld": "com"},
            "Telugu": {"code": "te", "tld": "co.in"},
            "Japanese": {"code": "ja", "tld": "co.jp"}
        }
        config_rules = lang_map.get(target_lang_selection, {"code": "es", "tld": "com"})
        
        tts = gTTS(
            text=translated_text, 
            lang=config_rules["code"], 
            tld=config_rules["tld"], 
            slow=False
        )
        
        timestamp = int(time.time())
        output_audio_filename = f"translated_speech_{timestamp}.mp3"
        tts.save(output_audio_filename)
        
        print(f"💾 Physical audio translation written successfully to disk: {output_audio_filename}")
        
        # Final Complete Successful Yield matches the parameters flawlessly!
        yield "✨ Translation Loop Complete!", raw_transcription, translated_text, output_audio_filename

    except Exception as e:
        print(f"❌ Voice Translation Pipeline Failure: {str(e)}")
        yield f"❌ System Error Encountered: {str(e)}", None, None, None


# ══════════════════════════════════════════════════════════════════
# GRADIO INTERFACE LAYOUT WINDOW CANVAS
# ══════════════════════════════════════════════════════════════════
with gr.Blocks(title="AI Voice-to-Voice Speech Translator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌐 AI Voice-to-Voice Speech Translator")
    gr.Markdown(
        "Record your voice or upload an audio file. This tool will transcribe the audio locally using your Mac's GPU, "
        "translate the text into a new language via your central model, and synthesize a downloadable speech audio file!"
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input component handles both live microphone inputs and uploaded file streams natively
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio Source")
            
            lang_dropdown = gr.Dropdown(
                choices=["Spanish", "French", "Hindi", "German", "Telugu", "Japanese"], 
                value="Spanish", 
                label="Target Translation Language"
            )
            
            submit_btn = gr.Button("Translate Speech", variant="primary")
            status_banner = gr.Textbox(value="Ready.", label="Pipeline Progress Status", interactive=False)
            
        with gr.Column(scale=2):
            source_text_box = gr.Textbox(label="1. Extracted Original Transcript Text (English Source)", interactive=False)
            translated_text_box = gr.Textbox(label="2. Generated Translated Text String", interactive=False)
            audio_output_player = gr.Audio(label="3. Synthesized Translated Speech Audio File Player", type="filepath")

    # Wire event listener mapping outputs matching the function's yield states exactly
    submit_btn.click(
        fn=run_voice_translation_pipeline,
        inputs=[audio_input, lang_dropdown],
        outputs=[status_banner, source_text_box, translated_text_box, audio_output_player]
    )

if __name__ == "__main__":
    print("🚀 Booting Voice-to-Voice Speech Translator interface on local loopback axis...")
    demo.launch(server_name="127.0.0.1", server_port=7862) # Launches on 7862 to avoid port congestion
