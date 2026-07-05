# CHANGE THESE OLD LINES:
# from config import Config
# from model import llm

# TO THESE EXPLICIT ABSOLUTE/RELATIVE CODES:
import sys
import os

# 1. Automatically find the path to your flask app folder dynamically
flask_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'genai_flask_app'))

# 2. Add that folder to Python's search path temporarily for this run
if flask_app_path not in sys.path:
    sys.path.append(flask_app_path)

# 3. Now you can safely import them without duplicating any code!
from config import Config
from model import llm


from langchain_litellm import ChatLiteLLM, LiteLLMEmbeddings

# Modern text processors and file handlers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# Modern LangChain compilation anchors
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

import gradio as gr
import os
import warnings

# Suppress runtime layout and package warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
warnings.filterwarnings('ignore')

## 1. Dynamic LLM Setup (Replaces WatsonxLLM)
""" def get_llm():
    # commenting as importing llm from model.py
    #Initializes a dynamic model based on your .env provider configurations.
    #Applies your custom course generations directly to the constructor parameters.
    
    # Combines to: "openai/gpt-4o-mini" or "anthropic/claude-3-5-sonnet-latest"
    full_model_string = f"{Config.LLM_PROVIDER}/{Config.MODEL_ID}"
    
    return ChatLiteLLM(
        model=full_model_string,
        temperature=0.5,    # Creative randomness parameter
        max_tokens=256      # Equivalent to MAX_NEW_TOKENS
    ) """

## 2. Dynamic Embedding Setup (Replaces WatsonxEmbeddings)
def get_embedding_model():
    """
    Leverages cloud-hosted API embeddings to avoid local package compilation.
    Matches your existing system token routes automatically.
    """
    # Uses OpenAI text-embedding-3-small as the lightweight cloud standard
    return LiteLLMEmbeddings(model="openai/text-embedding-3-small")

## 3. Document Loader (Slight fix to read file path string directly from Gradio)
def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

## 4. Text Splitter (Updated import path layout)
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_documents(data)

## 5. Vector Database (Swapped legacy community for modern langchain_chroma)
def vector_database(chunks):
    embedding_model = get_embedding_model()
    # Builds an in-memory instance of Chromadb dynamically for the session
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

## 6. Document Retriever Pipeline
def retriever(file_path):
    splits = document_loader(file_path)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever(search_kwargs={"k": 3})

## 7. QA Chain Process (Replaces RetrievalQA with modern document-stuffer)
def retriever_qa(file_path, query):
    if not file_path:
        return "Please upload a valid PDF document first."
        
    #llm = get_llm() -- Importing llm from model.py
    retriever_obj = retriever(file_path)
    
    # Fetch document text pieces matching the user query
    retrieved_docs = retriever_obj.invoke(query)
    
    # Establish modern explicit prompt structure
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Compile the simpler document manipulation chain 
    qa_chain = create_stuff_documents_chain(llm, prompt)
    
    # Invoke the execution payload mapping inputs directly
    response = qa_chain.invoke({
        "input": query,
        "context": retrieved_docs
    })
    
    return response

# 8. Create Gradio Interface (Fixed types to match modern filepath tracking)
rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        # Type 'filepath' passes a clean string path directly to your PyPDFLoader
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

# FIX: Completely remove 'allow_flagging="never"' from your Interface declaration deprecated


# Launch the dashboard locally
if __name__ == "__main__":
    print(f"🚀 Launching Gradio UI mapping requests to: {Config.LLM_PROVIDER.upper()}")
    rag_application.launch(server_name="127.0.0.1", server_port=7860)
