from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    """Base Flask configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", "default-fallback-key")
    #FLASK_ENV = os.getenv("FLASK_ENV", "production") # Not used here
    
    # AI/LLM Configurations
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Flexible model routing settings
    #LLM_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER", "openai").lower()
    #MODEL_ID = os.getenv("PRIMARY_TARGET_MODEL_NAME", "gpt-4o-mini")
    LLM_PROVIDER = os.getenv("SECONDARY_LLM_PROVIDER", "openai").lower()
    MODEL_ID = os.getenv("SECONDARY_TARGET_MODEL_NAME", "gpt-4o-mini")
    #LLM_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER").lower()
    #MODEL_ID = os.getenv("PRIMARY_TARGET_MODEL_NAME")
    #ANTHROPIC_MODEL_ID = os.getenv("TARGET_MODEL_NAME", "claude-3-5-sonnet-20240620")

    # ════════════════════════════════════════════════════════════════
    # 3. STYLE FINDER & IMAGE PROCESSING PARAMETERS
    # ════════════════════════════════════════════════════════════════
    # Dimensions required by standard TorchVision feature extractors (like ResNet/ViT)
    IMAGE_SIZE = (224, 224)
    
    # Image tensor transformation matrices 
    NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    NORMALIZATION_STD = [0.229, 0.224, 0.225]

    # ════════════════════════════════════════════════════════════════
    # 4. RAG & MATRIX VECTOR THRESHOLDS
    # ════════════════════════════════════════════════════════════════
    # Strictness modifier metric cutoff for Cosine Similarity search queries
    SIMILARITY_THRESHOLD = 0.8
    
    # Number of alternative style matches to return from a vector dataset lookup
    DEFAULT_ALTERNATIVES_COUNT = 5

class DevelopmentConfig(Config):
    DEBUG = True

#class ProductionConfig(Config):
#    DEBUG = False

    

    


