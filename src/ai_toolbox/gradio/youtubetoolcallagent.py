"""
Gradio Interface Dashboard for the Autonomous YouTube Agent Pipeline.
"""

import sys
import os
import re
import logging
import json
import warnings
from typing import List, Dict
import gradio as gr

# 1. Base LangChain and Tool Modules
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import Search
import yt_dlp

# 2. Path patch configuration to map your centralized configurations safely
current_script_dir = os.path.dirname(os.path.abspath(__file__))
flask_app_path = os.path.abspath(os.path.join(current_script_dir, '..', 'genai_flask_app'))
for route in [current_script_dir, flask_app_path]:
    if route not in sys.path:
        sys.path.append(route)

from config import Config
from model import initialize_any_model

warnings.filterwarnings("ignore")

# Suppress framework logger noise
for log_name in ['pytube', 'yt_dlp']:
    logging.getLogger(log_name).setLevel(logging.ERROR)

# Force the master agent engine to load OpenAI cloud pipelines to execute tool binding tasks natively
agent_llm = initialize_any_model(provider="openai", model_name="gpt-4o-mini")

# ══════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS 
# ══════════════════════════════════════════════════════════════════

@tool
def extract_video_id(url: str) -> str:
    """Extracts the 11-character YouTube video ID from a URL string."""
    pattern = r'(?:v=|be/|embed/|shorts/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else "Error: Invalid YouTube URL"


@tool
def fetch_transcript(video_id: str, language: str = "en") -> str:
    """Fetches the plain text transcript summary of a YouTube video based on its 11-character ID."""
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        return " ".join([segment['text'] for segment in transcript_data])
    except Exception:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            translated = transcript_list.find_transcript(['en']).translate(language)
            return " ".join([segment['text'] for segment in translated.fetch()])
        except Exception as e:
            return f"Error: Could not retrieve transcript: {str(e)}"


@tool
def search_youtube(query: str) -> List[Dict[str, str]]:
    """Search YouTube for videos matching the text query. Returns titles and video IDs."""
    try:
        s = Search(query)
        return [
            {
                "title": yt.title,
                "video_id": yt.video_id,
                "url": f"https://youtu.be{yt.video_id}"
            }
            for yt in s.results[:5]  # Limit bounds to top 5 results for token safety
        ]
    except Exception as e:
        return [{"error": f"Search execution failed: {str(e)}"}]


@tool
def get_full_metadata(url: str) -> dict:
    """Extract metadata given a YouTube URL, including views, duration, likes, and channel info."""
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title'),
                'views': info.get('view_count'),
                'duration': info.get('duration'),
                'channel': info.get('uploader'),
                'likes': info.get('like_count'),
                'comments': info.get('comment_count'),
                'chapters': info.get('chapters', [])
            }
    except Exception as e:
        return {"error": f"Failed to harvest metadata: {str(e)}"}


@tool
def get_thumbnails(url: str) -> List[Dict]:
    """Get available thumbnail resolutions for a YouTube video using its URL link."""
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            thumbnails = []
            for t in info.get('thumbnails', []):
                if 'url' in t:
                    thumbnails.append({
                        "url": t['url'],
                        "width": t.get('width'),
                        "height": t.get('height'),
                        "resolution": f"{t.get('width', '')}x{t.get('height', '')}".strip('x')
                    })
            return thumbnails
    except Exception as e:
        return [{"error": f"Failed to get thumbnails: {str(e)}"}]

# ══════════════════════════════════════════════════════════════════
# AGENT LOGIC ARRAYS
# ══════════════════════════════════════════════════════════════════

tools_list = [extract_video_id, fetch_transcript, search_youtube, get_full_metadata, get_thumbnails]

tool_mapping = {
    "extract_video_id": extract_video_id,
    "fetch_transcript": fetch_transcript,
    "search_youtube": search_youtube,
    "get_full_metadata": get_full_metadata,
    "get_thumbnails": get_thumbnails
}

llm_with_tools = agent_llm.bind_tools(tools_list)

def execute_tool(tool_call):
    tool_name = tool_call["name"]
    try:
        if tool_name in tool_mapping:
            result = tool_mapping[tool_name].invoke(tool_call["args"])
            content = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        else:
            content = f"Error: Tool '{tool_name}' is not supported."
    except Exception as e:
        content = f"Error: {str(e)}"
    return ToolMessage(content=content, tool_call_id=tool_call["id"])

# ══════════════════════════════════════════════════════════════════
# GRADIO CONVERSATION CHUTING LAYER
# ══════════════════════════════════════════════════════════════════

def run_agent_interface(user_query, chat_history_display):
    """
    Autonomous engine wrapped loop matching Gradio's strict dictionary message arrays.
    Updates progress messages in real-time as it jumps between reasoning steps.
    """
    if not user_query.strip():
        return "", chat_history_display

    if chat_history_display is None:
        chat_history_display = []

    # 1. Append the initial user prompt message to the screen dashboard
    chat_history_display.append({"role": "user", "content": user_query})
    yield "", chat_history_display

    try:
        # Build standard LangChain history list tracking frames
        messages_history = [HumanMessage(content=user_query)]
        
        # Initial call to the tool-aware model
        ai_response = llm_with_tools.invoke(messages_history)
        messages_history.append(ai_response)

        # Recursive Execution Loop Block
        max_iterations = 5
        iterations = 0
        
        while bool(getattr(messages_history[-1], 'tool_calls', None)) and iterations < max_iterations:
            last_msg = messages_history[-1]
            called_tools = [tc['name'] for tc in last_msg.tool_calls]
            
            # Post background actions status directly onto the UI timeline
            chat_history_display.append({
                "role": "assistant", 
                "content": f"⚙️ *System Agent Action: Invoking background tools: {', '.join(called_tools)}...*"
            })
            yield "", chat_history_display
            
            # Execute all requested tools in parallel
            tool_responses = [execute_tool(tc) for tc in last_msg.tool_calls]
            messages_history.extend(tool_responses)
            
            # Ask the LLM what to do next based on the tool data it just received
            next_ai_response = llm_with_tools.invoke(messages_history)
            messages_history.append(next_ai_response)
            iterations += 1

        # Extract final finalized synthesized text card
        final_text_report = messages_history[-1].content
        clean_markdown = final_text_report.replace("$", "\\$")

        # Post the finished report card cleanly to the screen portal 
        chat_history_display.append({"role": "assistant", "content": clean_markdown})
        yield "", chat_history_display

    except Exception as e:
        chat_history_display.append({"role": "assistant", "content": f"⚠️ Agent Loop Exception: {str(e)}"})
        yield "", chat_history_display


# ══════════════════════════════════════════════════════════════════
# GRADIO INTERFACE LAYOUT WINDOW CANVAS
# ══════════════════════════════════════════════════════════════════
with gr.Blocks(title="Autonomous YouTube Research Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Autonomous YouTube Research Agent")
    gr.Markdown(
        "Type a command to research YouTube. The agent will autonomously decide when to "
        "**search for videos**, **pull deep metadata matrices**, or **fetch and read caption transcripts** to answer your prompt!"
    )
    
    chatbot_canvas = gr.Chatbot(label="Agent Reasoning Terminal", height=500)
    
    with gr.Row():
        query_input = gr.Textbox(
            label="What would you like the agent to research?",
            placeholder="e.g., Search for 'iPhone 17 review' and write a short summary based on its video transcript.",
            scale=4
        )
        submit_btn = gr.Button("Execute Agent", variant="primary", scale=1)

    # Wire interface submit triggers (Supports click or text enter submission)
    submit_btn.click(
        fn=run_agent_interface,
        inputs=[query_input, chatbot_canvas],
        outputs=[query_input, chatbot_canvas]
    )
    query_input.submit(
        fn=run_agent_interface,
        inputs=[query_input, chatbot_canvas],
        outputs=[query_input, chatbot_canvas]
    )

if __name__ == "__main__":
    print("🚀 Launching Autonomous YouTube Agent Dashboard Portal...")
    # Safe port allocation mapping to bypass system conflicts
    demo.launch(server_name="127.0.0.1", server_port=7864, share=True)
