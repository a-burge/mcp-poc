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
        "gemini": "gemini-pro",
        "gpt5": "gpt-5-mini",
        # Add more models here as needed:
        # "gpt4": "gpt-4-turbo",
        # "claude": "claude-3-opus",
    }
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPIK_API_KEY: str = os.getenv("OPIK_API_KEY", "")
    OPIK_PROJECT_NAME: str = os.getenv("OPIK_PROJECT_NAME", "mcp-poc")
    
    # PDF Configuration
    PDF_URL: str = os.getenv("PDF_URL", "https://serlyfjaskra.is/example/smpc.pdf")
    
    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "300"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "0"))
    
    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", 
        "intfloat/multilingual-e5-large"
    )
    
    # Vector Store Configuration
    VECTOR_STORE_PATH: Path = Path(os.getenv("VECTOR_STORE_PATH", "data/vector_store"))
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    
    # Data directories
    DATA_DIR: Path = Path("data")
    PDFS_DIR: Path = DATA_DIR / "pdfs"
    RAW_SOURCE_DOCS_DIR: Path = DATA_DIR / "raw_source_docs"
    STRUCTURED_DIR: Path = DATA_DIR / "structured"
    
    # MCP Server Configuration
    MCP_SERVER_HOST: str = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", "8000"))
    
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
        cls.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
