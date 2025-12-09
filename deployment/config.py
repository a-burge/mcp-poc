"""Configuration management for MCP Server POC."""
import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""
    
    # LLM Provider Selection
    LLM_PROVIDER: Literal["gemini", "gpt5"] = os.getenv("LLM_PROVIDER", "gpt5").lower()
    
    # LLM Model Names - Centralized configuration for easy model switching
    LLM_MODELS: dict[str, str] = {
        "gemini": "gemini-2.5-flash",
        "gpt5": "gpt-5-mini",
        # Add more models here as needed:
        # "gpt4": "gpt-4-turbo",
        # "claude": "claude-3-opus",
    }
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    OPIK_API_KEY: str = os.getenv("OPIK_API_KEY", "")
    OPIK_PROJECT_NAME: str = os.getenv("OPIK_PROJECT_NAME", "mcp-poc")
    
    # PDF Configuration
    PDF_URL: str = os.getenv("PDF_URL", "https://serlyfjaskra.is/example/smpc.pdf")
    
    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", 
        "intfloat/multilingual-e5-large"
    )
    
    # Vector Store Configuration
    VECTOR_STORE_PATH: Path = Path(os.getenv("VECTOR_STORE_PATH", "data/vector_store"))
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "12"))  # Final results after re-ranking
    RETRIEVAL_INITIAL_K: int = int(os.getenv("RETRIEVAL_INITIAL_K", "12"))  # Initial retrieval before re-ranking (optimized: lowered from 20)
    RETRIEVAL_MULTI_MED_K: int = int(os.getenv("RETRIEVAL_MULTI_MED_K", "8"))  # Per medication for comparison queries
    RETRIEVAL_MIN_DOCS: int = int(os.getenv("RETRIEVAL_MIN_DOCS", "2"))  # Minimum docs threshold for fallback
    
    # Re-ranking Configuration
    ENABLE_RERANKING: bool = os.getenv("ENABLE_RERANKING", "false").lower() == "true"  # Enable re-ranking (default: False for speed)
    RERANKING_MODEL: str = os.getenv("RERANKING_MODEL", "gemini-2.5-flash")  # Cheaper/faster model for ranking (default: gemini-2.5-flash)
    RERANKING_DECISION_THRESHOLD: int = int(os.getenv("RERANKING_DECISION_THRESHOLD", "10"))  # Min docs to consider re-ranking
    
    # Query Rewrite Configuration
    ENABLE_QUERY_REWRITE: bool = os.getenv("ENABLE_QUERY_REWRITE", "false").lower() == "true"  # Enable query rewriting (default: False for safety)
    REWRITE_MODEL: str = os.getenv("REWRITE_MODEL", "gemini-2.5-flash")  # Model for query rewrite (default: gemini-2.5-flash)
    
    # Data directories
    DATA_DIR: Path = Path("data")
    PDFS_DIR: Path = DATA_DIR / "pdfs"
    RAW_SOURCE_DOCS_DIR: Path = DATA_DIR / "raw_source_docs"
    STRUCTURED_DIR: Path = DATA_DIR / "structured"
    ATC_DATA_DIR: Path = DATA_DIR / "atc"
    ATC_INDEX_PATH: Path = ATC_DATA_DIR / "atc_index.json"
    DRUG_ATC_MAPPINGS_PATH: Path = ATC_DATA_DIR / "drug_atc_mappings.json"
    INGREDIENTS_DATA_DIR: Path = DATA_DIR / "ingredients"
    INGREDIENTS_INDEX_PATH: Path = INGREDIENTS_DATA_DIR / "ingredients_index.json"
    
    # MCP Server Configuration
    MCP_SERVER_HOST: str = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", "8000"))
    
    # MCP Server Authentication
    MCP_AUTH_USERNAME: str = os.getenv("MCP_AUTH_USERNAME", "admin")
    MCP_AUTH_PASSWORD: str = os.getenv("MCP_AUTH_PASSWORD", "")
    
    @classmethod
    def get_mcp_server_url(cls) -> str:
        """Get MCP server URL, constructing from host and port if not set."""
        url = os.getenv("MCP_SERVER_URL", "")
        if url:
            return url
        return f"http://{cls.MCP_SERVER_HOST}:{cls.MCP_SERVER_PORT}"
    
    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        # Validate provider exists in model mapping
        if cls.LLM_PROVIDER not in cls.LLM_MODELS:
            valid_providers = ", ".join(cls.LLM_MODELS.keys())
            raise ValueError(
                f"Invalid LLM_PROVIDER: '{cls.LLM_PROVIDER}'. "
                f"Must be one of: {valid_providers}"
            )
        
        # Validate required API keys based on provider
        if cls.LLM_PROVIDER == "gemini" and not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is required when LLM_PROVIDER is 'gemini'"
            )
        if cls.LLM_PROVIDER == "gpt5" and not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER is 'gpt5'"
            )
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.PDFS_DIR.mkdir(exist_ok=True)
        cls.RAW_SOURCE_DOCS_DIR.mkdir(exist_ok=True)
        cls.STRUCTURED_DIR.mkdir(exist_ok=True)
        cls.ATC_DATA_DIR.mkdir(exist_ok=True)
        cls.INGREDIENTS_DATA_DIR.mkdir(exist_ok=True)
        cls.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
