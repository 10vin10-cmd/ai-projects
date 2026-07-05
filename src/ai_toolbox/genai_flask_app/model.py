from langchain_core.prompts import ChatPromptTemplate
from langchain_litellm import ChatLiteLLM
from config import Config  # Import the Config class

#Output
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

"""
Unified Model Initialization Layer providing explicit Cloud and Local LLM instantiation hooks.
"""
import os
import sys
from config import Config

# Modernized LangChain Unified Provider Packages
#from langchain_litellm.chat_models import ChatLiteLLM
from langchain_ollama import ChatOllama

# ══════════════════════════════════════════════════════════════════
# METHOD 1: DEDICATED CLOUD INITIALIZER (OPENAI / CLAUDE / ANTHROPIC)
# ══════════════════════════════════════════════════════════════════
def get_cloud_model(provider: str = None, model_name: str = None):
    """
    Explicitly instantiates an enterprise cloud LLM via LiteLLM.
    """
    target_provider = str(provider or Config.LLM_PROVIDER).lower().strip()
    target_model = str(model_name or Config.MODEL_ID).strip()
    
    if "/" not in target_model and target_provider:
        full_model_string = f"{target_provider}/{target_model}"
    else:
        full_model_string = target_model
        
    print(f"☁️ Cloud Model Active: Initializing '{full_model_string}'...")
    return ChatLiteLLM(
        model=full_model_string,
        temperature=0.2,       
        max_tokens=2000
    )

# ══════════════════════════════════════════════════════════════════
# METHOD 2: DEDICATED LOCAL INITIALIZER (LLAMA VIA OLLAMA)
# ══════════════════════════════════════════════════════════════════
def get_local_model(model_name: str = "llama3.2"):
    """
    Explicitly instantiates a 100% free, private local model running via Ollama.
    ✅ FIX: Removed all custom raw hyperparameter keyword properties entirely.
    This safely prevents all deep Client.chat() signature tracking mismatches!
    """
    target_model = str(model_name).strip()
    print(f"🏠 Local Model Active: Initializing '{target_model}' via Ollama pipeline...")
    
    # Passing only the canonical model name is the only 100% bulletproof method
    # that functions across every single LangChain environment update version.
    return ChatOllama(model=target_model)

# ══════════════════════════════════════════════════════════════════
# METHOD 3: THE AUTOMATED CENTRAL FACTORY LOOKUP
# ══════════════════════════════════════════════════════════════════
def initialize_any_model(provider: str, model_name: str):
    """
    Dynamic fallback router that switches between Method 1 and Method 2 
    natively based on the provider keyword.
    """
    provider_clean = str(provider).lower().strip()
    
    if provider_clean in ["ollama", "local"]:
        return get_local_model(model_name=model_name)
    else:
        return get_cloud_model(provider=provider, model_name=model_name)

# ══════════════════════════════════════════════════════════════════
# AUTOMATED MASTER INSTANTIATION AXIS
# ══════════════════════════════════════════════════════════════════
llm = initialize_any_model(
    provider=Config.LLM_PROVIDER, 
    model_name=Config.MODEL_ID
)


# OPTIMIZED: Structured chat messages instead of hardcoded raw string tokens
chat_template = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}\n\n{format_prompt}"),
    ("user", "{user_prompt}")
])

# Define JSON output structure
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    response: str = Field(description="Suggested response to the user")

# JSON output parser
json_parser = JsonOutputParser(pydantic_object=AIResponse)

def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model | json_parser
    parsed_output = chain.invoke({
        'system_prompt': system_prompt, 
        'user_prompt': user_prompt, 
        'format_prompt': json_parser.get_format_instructions()
    })
    
    # FIX: Do NOT return parsed_output.content. 
    # 'parsed_output' is already a clean Python dictionary matching your Pydantic schema!
    return parsed_output 

# Alternative definition inside model.py
#structured_llm = llm.with_structured_output(AIResponse)

def get_ai_response_modern(model, system_prompt, user_prompt):
    """
    Leverages native structured output mapping. 
    Removes the need for a manual prompt template pipeline object.
    """
    # Bind the Pydantic schema directly to the model instance
    structured_llm = model.with_structured_output(AIResponse)
    
    # Construct standard chat format tuples
    messages = [
        ("system", system_prompt),
        ("user", user_prompt)
    ]
    # The response will be an actual instantiated instance of your AIResponse Pydantic class!
    return structured_llm.invoke(messages)

