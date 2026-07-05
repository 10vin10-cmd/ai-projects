"""
AI Knowledge Base RAG Tutor — Advanced Local Storage Vector Caching System.
"""
"""
AI Knowledge Base RAG Tutor — Standalone Local Ollama Integration.
"""

import sys
import os
import time
import warnings
import concurrent.futures
import hashlib  
import shutil  
import gradio as gr

# Modernized LangChain v1.0 layout dependencies
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader, OnlinePDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from pdf2image import convert_from_path
import pytesseract

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# ✅ PASTE THE SHIELD CODE RIGHT HERE (STARTING AROUND LINE 26)
# ══════════════════════════════════════════════════════════════════
import ollama

original_ollama_chat = ollama.chat

def safe_ollama_chat_wrapper(*args, **kwargs):
    """
    Safely intercepts all low-level data dictionary inputs.
    Removes the conflicting 'temperature' key if LiteLLM attempts to inject it!
    """
    if 'temperature' in kwargs:
        kwargs.pop('temperature')
    if 'options' in kwargs and isinstance(kwargs['options'], dict) and 'temperature' in kwargs['options']:
        kwargs['options'].pop('temperature')
        
    return original_ollama_chat(*args, **kwargs)

ollama.chat = safe_ollama_chat_wrapper
print("🛡️ System Shield Active: Low-level Ollama parameter interceptor initialized successfully.")
# ══════════════════════════════════════════════════════════════════


current_dir = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(current_dir, "tutor_vector_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class TutorSessionState:
    def __init__(self):
        self.vector_db = None
        self.chat_history = []

session = TutorSessionState()

# This initialization step is now fully shielded and safe!
#local_llm = ChatOllama(model="llama3.2")

# ... (The rest of your script code continues normally below this)

#Path patch configuration to map your centralized configurations
current_dir = os.path.dirname(os.path.abspath(__file__))
flask_app_path = os.path.abspath(os.path.join(current_dir, '..', 'genai_flask_app'))
if flask_app_path not in sys.path:
    sys.path.append(flask_app_path)

from config import Config
from model import llm  # Your active central ChatLiteLLM instance

from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader, OnlinePDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from pdf2image import convert_from_path
import pytesseract

warnings.filterwarnings('ignore')

# NEW: Define a dedicated storage directory for saved vector files
CACHE_DIR = os.path.join(current_dir, "tutor_vector_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class TutorSessionState:
    def __init__(self):
        self.vector_db = None
        self.chat_history = []

session = TutorSessionState()

from langchain_ollama import ChatOllama
# ✅ DIRECT INITIALIZATION: Instantiate Ollama directly with zero global dependencies
# This forces LangChain to use pure defaults and drops the 'temperature' parameter completely!
local_llm = ChatOllama(model="llama3.2")

def get_cache_path(input_source, is_url=False):
    """
    Generates a safe, repeatable folder path for storing the FAISS index.
    """
    if is_url:
        # Generate an MD5 hash string of the URL to prevent illegal folder characters
        url_hash = hashlib.md5(input_source.encode('utf-8')).hexdigest()
        return os.path.join(CACHE_DIR, f"web_{url_hash}")
    else:
        # Use the raw filename of the uploaded PDF
        base_name = os.path.basename(input_source).replace(" ", "_")
        return os.path.join(CACHE_DIR, f"pdf_{base_name}")


def compile_knowledge_base(input_source, is_url=False):
    """
    Loads text content, running local OCR if needed, and builds a FAISS index.
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings_engine = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if is_url:
            source_str = str(input_source).strip()
            if source_str.lower().endswith('.pdf'):
                loader = OnlinePDFLoader(source_str)
                raw_documents = loader.load()
            else:
                loader = WebBaseLoader(
                    web_path=source_str,
                    header_template={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                        "Accept-Language": "en-US,en;q=0.9"
                    }
                )
                raw_documents = loader.load()
        else:
            loader = PyMuPDFLoader(input_source)
            raw_documents = loader.load()
            
            # Automated OCR Rescue Layer for Scanned Pages
            total_extracted_chars = sum([len(doc.page_content.strip()) for doc in raw_documents])
            if total_extracted_chars < 100:
                print("⚙️ Scanned document detected. Running OCR processing...")
                pdf_pages = convert_from_path(input_source, dpi=150)
                ocr_documents = []
                for idx, page_image in enumerate(pdf_pages):
                    page_text = pytesseract.image_to_string(page_image)
                    ocr_documents.append(Document(
                        page_content=page_text,
                        metadata={"source": input_source, "page": idx + 1}
                    ))
                raw_documents = ocr_documents
        
        if not raw_documents:
            return "INDEX_ERROR: Target source returned a completely blank layout map."
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(raw_documents)
        
        vector_index = FAISS.from_documents(split_docs, embeddings_engine)
        return vector_index
    except Exception as e:
        return f"INDEX_ERROR: {str(e)}"


def ingest_study_material(pdf_file, url_input):
    """
    Orchestrates the ingestion lifecycle. Checks the local hard drive 
    for cached indices to avoid unneeded computations.
    """
    is_url = False
    source_target = None
    
    if pdf_file is not None:
        source_target = pdf_file.name
        is_url = False
    elif url_input and url_input.strip() != "":
        source_target = url_input.strip()
        is_url = True
        
    if not source_target:
        return "⚠️ Ingestion Fault: Provide an input first!", gr.update(interactive=False)
        
    try:
        # 1. NEW: Calculate the unique target caching location path
        cache_folder = get_cache_path(source_target, is_url)
        
        # 2. NEW: Check if the vector directory files exist on your Mac disk
        if os.path.exists(cache_folder):
            print(f"💾 Cache Hit! Loading pre-computed index instantly from: {cache_folder}")
            yield "⚡ Fast-loading saved structural index from your local Mac storage...", gr.update(interactive=False)
            
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings_engine = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Load the index natively off disk in 0.1 seconds, allowing untrusted files
            session.vector_db = FAISS.load_local(cache_folder, embeddings_engine, allow_dangerous_deserialization=True)
            session.chat_history = []
            
            yield "🚀 Index loaded from disk cache! Ask your tutor anything below.", gr.update(interactive=True)
            return

        # 3. Cache Miss: Execute regular heavy thread index creation
        yield "⏳ First-time run: Parsing asset and computing vector embeddings...", gr.update(interactive=False)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(compile_knowledge_base, source_target, is_url)
            while not future.done():
                time.sleep(0.1)
            result = future.result()
            
        if isinstance(result, str) and "INDEX_ERROR" in result:
            raise Exception(result.replace("INDEX_ERROR: ", ""))
            
        # 4. NEW: Permanently save the fresh index results straight to your disk
        print(f"💾 Saving newly computed vector matrix index to: {cache_folder}")
        result.save_local(cache_folder)
        
        session.vector_db = result
        session.chat_history = []  
        
        yield "✅ Knowledge Base successfully compiled and cached! Ask your tutor below.", gr.update(interactive=True)
        
    except Exception as e:
        print(f"❌ PDF Ingestion Failure: {str(e)}")
        yield f"❌ Failed to parse resource: {str(e)}", gr.update(interactive=False)


def tutor_student_loop(student_question, chat_history_display):
    """Traverses local vectors and streams Socratic lesson responses."""
    if not student_question.strip():
        return "", chat_history_display
    if not session.vector_db:
        raise gr.Error("Please index a resource first!")

    try:
        retriever = session.vector_db.as_retriever(search_kwargs={"k": 4})
        matched_chunks = retriever.invoke(student_question)
        context_block = "\n\n".join([doc.page_content for doc in matched_chunks])
        
        system_prompt = """You are an expert Personal Tutor. Teach using the resource context below. Keep answers concise (under 3 paragraphs) to match token limits. End with an interactive question.
        CONTEXT:\n{context}"""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        tutor_chain = prompt_template | llm.bind(temperature=0.2) | StrOutputParser()
        tutor_response = tutor_chain.invoke({"context": context_block, "history": session.chat_history, "question": student_question})
        
        clean_response = tutor_response.replace("$", "\\$")
        session.chat_history.append(HumanMessage(content=student_question))
        session.chat_history.append(AIMessage(content=clean_response))
        
        if chat_history_display is None:
            chat_history_display = []
        chat_history_display.append({"role": "user", "content": student_question})
        chat_history_display.append({"role": "assistant", "content": clean_response})
        
        return "", chat_history_display
    except Exception as e:
        chat_history_display.append({"role": "user", "content": student_question})
        chat_history_display.append({"role": "assistant", "content": f"⚠️ Error: {str(e)}"})
        return "", chat_history_display


# ══════════════════════════════════════════════════════════════════
# GRADIO STUDY TUTOR WEB INTERFACE LAYOUT
# ══════════════════════════════════════════════════════════════════
with gr.Blocks(title="AI Socratic Knowledge Tutor Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 AI Socratic Knowledge Tutor Dashboard (OCR Pro Edition)")
    gr.Markdown(
        "Upload a local PDF textbook (supports selectable text or flat scanned images). "
        "Our updated pipeline runs automated OCR to harvest characters, map vectors, and open up an interactive learning terminal canvas!"
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Step 1: Input Learning Material Source")
            pdf_uploader = gr.File(label="Option A: Upload Local PDF File", file_types=[".pdf"])
            url_input = gr.Textbox(
                label="Option B: Paste Online Resource URL Link", 
                placeholder="e.g., https://wikipedia.org"
            )
            ingest_btn = gr.Button("Index Study Materials", variant="primary")
            indexing_status = gr.Textbox(value="Awaiting inputs...", label="Indexer Status Logs", interactive=False)
            
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Step 2: Interactive Tutoring Session Canvas")
            chatbot_canvas = gr.Chatbot(label="Tutor Discussion Board Log", height=450)
            with gr.Row():
                student_input = gr.Textbox(
                    label="Ask your tutor a question about the study materials...", 
                    placeholder="e.g., Can you explain the main concepts of this chapter?",
                    scale=4,
                    interactive=False
                )
                ask_btn = gr.Button("Submit Question", variant="primary")
            

    # 1. Wire Document Ingestion Listener Action
    ingest_btn.click(
        fn=ingest_study_material,
        inputs=pdf_uploader,
        outputs=[indexing_status, student_input]
    )
    
    # 2. Wire Chat Interaction Submit Triggers (Supports clicking button or pressing Enter key)
    ask_btn.click(
        fn=tutor_student_loop,
        inputs=[student_input, chatbot_canvas],
        outputs=[student_input, chatbot_canvas]
    )
    student_input.submit(
        fn=tutor_student_loop,
        inputs=[student_input, chatbot_canvas],
        outputs=[student_input, chatbot_canvas]
    )

    if __name__ == "__main__":
        # Launch safely outside of standard system port bottleneck tracks
        print("🚀 Initializing Socratic Tutor application system interface...")
        demo.launch(server_name="127.0.0.1", server_port=7863)
