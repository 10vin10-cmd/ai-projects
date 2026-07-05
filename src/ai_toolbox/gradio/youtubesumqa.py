import gradio as gr
import faiss
import os
import sys
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into manageable segments
from langchain_core.prompts import ChatPromptTemplate  # For defining prompt templates
from langchain_core.output_parsers import StrOutputParser

# 1. Step up one directory layer out of 'vector', then step into 'genai_flask_app'
flask_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'genai_flask_app'))

# 2. Append this folder location dynamically to Python's file routing engine
if flask_app_path not in sys.path:
    sys.path.append(flask_app_path)

from config import Config
from model import initialize_any_model, llm



print("✅ FAISS engine initialized successfully.")
print("✅ YouTube Transcript API linked.")
print("✅ Streamlit UI environment loaded.")

def get_video_id(url):    
    # Regex pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
video_id = get_video_id(url)
print(video_id)  # Output: dQw4w9WgXcQ

def get_transcript(url):
    # Extracts the video ID from the URL
    video_id = get_video_id(url)
    
    # Create a YouTubeTranscriptApi() object
    ytt_api = YouTubeTranscriptApi()
    
    # Fetch the list of available transcripts for the given YouTube video
    transcripts = ytt_api.list(video_id)
    
    transcript = ""
    for t in transcripts:
        # Check if the transcript's language is English
        if t.language_code == 'en':
            if t.is_generated:
                # If no transcript has been set yet, use the auto-generated one
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                # If a manually created transcript is found, use it (overrides auto-generated)
                transcript = t.fetch()
                break  # Prioritize the manually created transcript, exit the loop
    
    return transcript if transcript else None

# Sample YouTube URL
url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Fetching the transcript
transcript = get_transcript(url)

# Output the fetched transcript
print(transcript)

def process(transcript):
    # Initialize an empty string to hold the formatted transcript
    txt = ""
    
    # Loop through each entry in the transcript
    for i in transcript:
        try:
            # Append the text and its start time to the output string
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            # If there is an issue accessing 'text' or 'start', skip this entry
            pass
            
    # Return the processed transcript as a single string
    return txt

# Processing the transcript
formatted_transcript = process(transcript)

# Output the processed transcript
print(formatted_transcript)



def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    """
    Splits a raw YouTube transcript string cleanly into structured document chunks
    ready for FAISS vector indexing.
    """
    # Initialize the character splitter layout
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # FIX: Use create_documents to turn the raw text string into a list of Document objects
    chunks = text_splitter.create_documents([processed_transcript])
    
    print(f"✂️ Transcript split successfully into {len(chunks)} text chunks.")
    return chunks


# Sample processed transcript string
processed_transcript = """Text: We're no strangers to love. Start: 0.0
Text: You know the rules and so do I. Start: 3.5
Text: A full commitment's what I'm thinking of. Start: 7.5"""

# Chunking the transcript
chunks = chunk_transcript(processed_transcript)

# Output the chunks
print(chunks)


def create_faiss_index(chunks, llm):
    """
    Create a FAISS index from text chunks using the specified embedding model.
    
    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    <span data-type="emoji" data-name="return"></span> FAISS index
    """
    # Use the FAISS library to create an index from the provided text chunks
    return faiss.from_texts(chunks, llm)

def perform_similarity_search(faiss_index, query, k=3):
    """
    Search for specific queries within the embedded transcript using the FAISS index.
    
    :param faiss_index: The FAISS index containing embedded text chunks
    :param query: The text input for the similarity search
    :param k: The number of similar results to return (default is 3)
    <span data-type="emoji" data-name="return"></span> List of similar results
    """
    # Perform the similarity search using the FAISS index
    results = faiss_index.similarity_search(query, k=k)
    return results

def create_summary_prompt():
    """
    Create a PromptTemplate for summarizing a YouTube video transcript.
    
    <span data-type="emoji" data-name="return"></span> PromptTemplate object
    """
    # Define the template for the summary prompt
    system_instructions = """

    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:
    """
    
    # Create the PromptTemplate object with the defined template
 

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instructions),
        ("human", "Please summarize this text segment:\n\n{transcript}")
    ])
    return prompt

def create_summary_chain(llm, prompt, verbose=True):
    """
    Create a modern LCEL pipeline for generating summaries.
    Replaces the legacy, deprecated LLMChain class.
    
    :param llm: Dynamic ChatLiteLLM instance
    :param prompt: ChatPromptTemplate or PromptTemplate instance
    :param verbose: Kept for signature compatibility (LCEL uses callbacks for logging)
    :return: A runnable LCEL chain sequence
    """
    # Simply pipe the components together. 
    # StrOutputParser ensures the chain outputs a clean plain-text string natively.
    return prompt | llm | StrOutputParser()

def retrieve(query, faiss_index, k=7):
    """
    Retrieve relevant context from the FAISS index based on the user's query.

    Parameters:
        query (str): The user's query string.
        faiss_index (FAISS): The FAISS index containing the embedded documents.
        k (int, optional): The number of most relevant documents to retrieve (default is 3).

    Returns:
        list: A list of the k most relevant documents (or document chunks).
    """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context

from langchain_core.prompts import ChatPromptTemplate

def create_chat_qa_prompt_template():
    """
    Create a highly structured ChatPromptTemplate optimized for modern Chat models.
    """
    return ChatPromptTemplate.from_messages([
        (
            "system", 
            "You are an expert assistant providing detailed answers based on video content.\n\n"
            "Relevant Video Context:\n{context}"
        ),
        (
            "human", 
            "Based on the context provided, please answer this question:\n{question}"
        )
    ])

def create_qa_chain(llm, prompt):

    """
    Create a modern LCEL pipeline for question-answering.
    Replaces the legacy, deprecated LLMChain class.
    
    :param llm: Dynamic ChatLiteLLM instance
    :param prompt: ChatPromptTemplate or PromptTemplate instance
    :return: A runnable LCEL chain sequence
    """
    # Simply pipe the components together. 
    # StrOutputParser ensures the chain outputs a clean plain-text string natively.
    return prompt | llm | StrOutputParser()

def generate_answer(question, faiss_index, qa_chain, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.

    Args:
        question: str
            The user's question.
        faiss_index: FAISS
            The FAISS index containing the embedded documents.
        qa_chain: LLMChain
            The question-answering chain (LLMChain) to use for generating answers.
        k: int, optional (default=3)
            The number of relevant documents to retrieve.

    Returns:
        str: The generated answer to the user's question.
    """

    # 3. Invoke it seamlessly with your data dictionary
    answer = qa_chain.invoke({"context": relevant_context, "question": question})
    return answer



# Initialize an empty string to store the processed transcript after fetching and preprocessing
processed_transcript = ""

import sys
import os

# Ensure you have your global/imported functions ready
# (Assuming get_transcript, process, create_summary_prompt, and create_summary_chain exist)

def summarize_video(video_url):
    """
    Title: Summarize Video

    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.

    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global fetched_transcript, processed_transcript
    
    # FIX: Corrected indentation alignment to standard 4-spaces to prevent crash
    if video_url:
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."

    if processed_transcript:
        # Create the summary prompt and chain
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        # FIX: Swapped out the deprecated .run() for the modern .invoke()
        # This aligns with the LCEL prompt | llm | StrOutputParser() setup
        summary = summary_chain.invoke({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."

    
def answer_question(video_url, user_question):
    """
    Title: Answer User's Question

    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query 
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.

    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
    """
    global fetched_transcript, processed_transcript

    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."

    if processed_transcript and user_question:
        # Step 1: Chunk the transcript (only for Q&A)
        chunks = chunk_transcript(processed_transcript)

        # Step 4: Create FAISS index for transcript chunks (only needed for Q&A)
        faiss_index = create_faiss_index(chunks, llm)

        # Step 5: Set up the Q&A prompt and chain
        qa_prompt = create_chat_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        # Step 6: Generate the answer using FAISS index
        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."

with gr.Blocks() as interface:
    # Input field for YouTube URL
    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")
    
    # Outputs for summary and answer
    summary_output = gr.Textbox(label="Video Summary", lines=5)
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

    # Buttons for selecting functionalities after fetching transcript
    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    # Display status message for transcript fetch
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    # Set up button actions
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

# Launch the app with specified server name and port
interface.launch(server_name="0.0.0.0", server_port=7860)
