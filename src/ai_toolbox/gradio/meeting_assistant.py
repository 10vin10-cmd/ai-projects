import os
import sys
import warnings
import torch
import gradio as gr
from transformers import pipeline

# 1. Dynamically append cross-folder routing paths to look for your master config setup
flask_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'genai_flask_app'))
if flask_app_path not in sys.path:
    sys.path.append(flask_app_path)

from config import Config
from model import initialize_any_model

# Modernized core LangChain v1.0 imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Suppress Python 3.14 layout and package migration warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# LLM ENGINE INITIALIZATION (Replaces legacy WatsonxLLM configuration)
# ------------------------------------------------------------------
# This automatically spins up your configured model (e.g. OpenAI/Anthropic) using your .env keys
llm = initialize_any_model(
    provider=Config.LLM_PROVIDER,
    model_name=Config.MODEL_ID
)

# Apply your custom course parameter tokens natively to the constructor properties
llm.max_tokens = 512
llm.temperature = 0.5

# ------------------------------------------------------------------
# HELPER FUNCTIONS & AGENT ACTIONS
# ------------------------------------------------------------------
def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)

def product_assistant(ascii_transcript):
    """
    Acts as a specialized financial product terminology validation layer.
    Replaces the legacy ModelInference logic with your active modern engine.
    """
    system_prompt = """You are an intelligent assistant specializing in financial products;
    your task is to process transcripts of earnings calls, ensuring that all references to
    financial products and common financial terms are in the correct format. For each
    financial product or common term that is typically abbreviated as an acronym, the full term 
    should be spelled out followed by the acronym in parentheses. For example, '401k' should be
    transformed to '401(k) retirement savings plan', 'HSA' should be transformed to 'Health Savings Account (HSA)' , 'ROA' should be transformed to 'Return on Assets (ROA)', 'VaR' should be transformed to 'Value at Risk (VaR)', and 'PB' should be transformed to 'Price to Book (PB) ratio'. Similarly, transform spoken numbers representing financial products into their numeric representations, followed by the full name of the product in parentheses. For instance, 'five two nine' to '529 (Education Savings Plan)' and 'four zero one k' to '401(k) (Retirement Savings Plan)'. However, be aware that some acronyms can have different meanings based on the context (e.g., 'LTV' can stand for 'Loan to Value' or 'Lifetime Value'). You will need to discern from the context which term is being referred to and apply the appropriate transformation. In cases where numerical figures or metrics are spelled out but do not represent specific financial products (like 'twenty three percent'), these should be left as is. Your role is to analyze and adjust financial product terminology in the text. Once you've done that, produce the adjusted transcript and a list of the words you've changed"""
    
    # Structure the message explicitly using a modern ChatPromptTemplate array layout
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{transcript}")
    ])
    
    # Dynamically bind the specific low-temperature overrides required for accuracy task chains
    validation_chain = agent_prompt | llm.bind(temperature=0.2)
    #validation_chain.bound.temperature = 0.2
    
    response = validation_chain.invoke({"transcript": ascii_transcript})
    return response.content

# ------------------------------------------------------------------
# PROMPT TEMPLATE AND LCEL CHAIN COMPILATION
# ------------------------------------------------------------------
template = """Generate meeting minutes and a list of tasks based on the provided context.

Context:
{context}

Meeting Minutes:
Key points discussed
Decisions made

Task List:
Actionable items with assignees and deadlines

Detailed Output:"""

# Compiled via modern, lightning-fast LangChain Expression Language (LCEL) pipes
prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"context": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ------------------------------------------------------------------
# AUDIO SPEECH-TO-TEXT PROCESSING PIPELINE
# ------------------------------------------------------------------
def transcript_audio(audio_file):
    if not audio_file:
        return "Please upload an audio file.", None

    # Hardware Acceleration Hook: Routes calculations to your Mac's integrated GPU cores
    target_device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Initialize the modern local Whisper Speech-To-Text pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        # Chaning to MAC OPTIMIZED DISTIL SLUG instead of the original OpenAI Whisper model for faster inference on Apple Silicon:
        model="distil-whisper/distil-medium.en",
        #model="openai/whisper-medium",
        chunk_length_s=30,
        device=target_device  # Optimizes runtime on your MacBook Air
    )
    
    print("🎙️ Local Whisper Engine: Transcribing meeting audio recording...")
    raw_transcript = pipe(audio_file, batch_size=8)["text"]
    ascii_transcript = remove_non_ascii(raw_transcript)
    
    print("🧠 Terminology Layer: Formatting financial abbreviations...")
    adjusted_transcript = product_assistant(ascii_transcript)
    
    print("📝 Aggregation Chain: Documenting structural meeting minutes and action items...")
    result = chain.invoke(adjusted_transcript)
    
    # Write the result to a physical local file for user download
    output_file = "meeting_minutes_and_tasks.txt"
    with open(output_file, "w") as file:
        file.write(result)
        
    print("✅ Pipeline execution complete. Outputs generated successfully.")
    return result, output_file

# ------------------------------------------------------------------
# MODERNIZED GRADIO USER INTERFACE LAYOUT
# ------------------------------------------------------------------
# FIX 1: Passed source input parameter as a list array mapping format
audio_input = gr.Audio(sources=["upload"], type="filepath", label="Upload your audio file")
output_text = gr.Textbox(label="Meeting Minutes and Tasks")
download_file = gr.File(label="Download the Generated Meeting Minutes and Tasks")

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=[output_text, download_file],
    title="AI Meeting Assistant",
    description="Upload an audio file of a meeting. This tool will transcribe the audio, fix product-related terminology, and generate meeting minutes along with a list of tasks."
)

# Launch the app from an open, safe, non-crashing local port axis
if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860)
