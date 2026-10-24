# ai-projects

Hands-on GenAI and data-engineering work: RAG pipelines, LLM tool-calling agents, vector search, speech and vision apps, and SQL/Python practice. Most apps run locally against Ollama or a hosted model through a shared config layer.

## What's inside

| Area | Path | Highlights |
|---|---|---|
| **RAG & document Q&A** | [`src/ai_toolbox/gradio/pdf_tutor.py`](src/ai_toolbox/gradio/pdf_tutor.py) | PDF tutor: PyMuPDF + OCR (Tesseract) ingestion, chunking, FAISS index, LangChain chain on a local Ollama model, Gradio UI, parallel page processing with content-hash caching |
| **Vector search** | [`src/ai_toolbox/vector/`](src/ai_toolbox/vector) | Semantic and filtered search, result limiting, retrieval-system comparison, a RAG chatbot over a structured food dataset |
| **LLM apps (Gradio)** | [`src/ai_toolbox/gradio/`](src/ai_toolbox/gradio) | Meeting-minutes assistant (speech → transcript → action items), voice translator, image captioning, YouTube summarise-and-Q&A, tool-calling agent |
| **Web app** | [`src/ai_toolbox/genai_flask_app/`](src/ai_toolbox/genai_flask_app) | Flask GenAI app with model abstraction and config layer |
| **Calorie coach** | [`cal_coach_app/`](cal_coach_app) | Flask app with HTML front end |
| **LinkedIn icebreaker** | [`icebreaker/`](icebreaker) | LlamaIndex + watsonx embeddings over profile data, CLI and Gradio UI |
| **Agent & retrieval labs** | [`JL/`](JL) | Notebooks: manual tool-calling agents, LCEL pipelines, LlamaIndex advanced retrievers, FAISS similarity search, text-to-chart agent |
| **LLM internals** | [`logprob.ipynb`](logprob.ipynb), [`apicosts.ipynb`](apicosts.ipynb) | Token log-probabilities and API cost comparison across models |
| **SQL / data practice** | [`Arch/`](Arch) | Window-function queries (rolling averages), JSON flattening |
| **CI/CD** | [`.github/workflows/`](.github/workflows) | Python package tests; OCI Resource Manager apply with retry for free-tier capacity |

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest
```

Individual apps list their own dependencies (see `icebreaker/requirements.txt`). Model keys are read from a local `.env`, which is git-ignored and never committed.

## Stack

Python · LangChain · LlamaIndex · FAISS · ChromaDB · Ollama · OpenAI / watsonx · Gradio · Flask · Tesseract · GitHub Actions

Sample data files (`FoodDataSet.json`, `companypolicies.txt`, `meeting_minutes_and_tasks.txt`, `sample-meeting.wav`) are synthetic course datasets used by the demos.
