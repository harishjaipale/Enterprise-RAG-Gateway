import logging
from typing import Optional, Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("ConfigurationEngine")


class QdrantSettings(BaseSettings):
    """Qdrant Vector Database configuration settings."""
    url: str = Field(
        default="http://localhost:6333",
        description="Qdrant database endpoint URL."
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key for cloud or secured clusters."
    )
    collection_name: str = Field(
        default="enterprise_rag_vector_index",
        description="Target Qdrant vector index collection name."
    )


class LLMSettings(BaseSettings):
    """LLM provider authentication keys and operational configuration."""
    # Added "groq" to provider options
    provider: Literal["openai", "groq", "anthropic", "cohere", "ollama", "local"] = Field(
        default="openai",
        description="Active primary LLM inference provider."
    )
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key.")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key for fast/fallback LLM inference.")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key.")
    cohere_api_key: Optional[str] = Field(default=None, description="Cohere API key.")
    
    # Local/Ollama Fallback Support
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama local service base endpoint."
    )

    @model_validator(mode="after")
    def validate_provider_keys(self) -> "LLMSettings":
        """Ensures active provider key is present without crashing on local setups."""
        # Allow local/ollama providers without commercial keys
        if self.provider in ["ollama", "local"]:
            return self
            
        # Check commercial API keys (Added groq_api_key in check)
        if not any([self.openai_api_key, self.groq_api_key, self.anthropic_api_key, self.cohere_api_key]):
            logger.warning(
                "No commercial LLM API Key detected in environment. "
                "Defaulting operational mode to local/fallback LLM engines."
            )
        return self


class ChunkingSettings(BaseSettings):
    """Document ingestion chunking and splitting parameters."""
    chunk_size: int = Field(
        default=1024, 
        gt=0, 
        le=8192, 
        description="Document chunk size in tokens/words."
    )
    chunk_overlap: int = Field(
        default=200, 
        ge=0, 
        description="Token overlap between adjacent chunks."
    )

    @model_validator(mode="after")
    def validate_overlap_bounds(self) -> "ChunkingSettings":
        """Ensures chunk overlap is strictly smaller than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"Chunk overlap ({self.chunk_overlap}) must be strictly less than chunk size ({self.chunk_size})!"
            )
        return self


class ParserSettings(BaseSettings):
    """Document parser engine options."""
    llama_cloud_api_key: Optional[str] = Field(
        default=None,
        description="LlamaParse cloud API key for layout extraction."
    )
    parsing_tier: str = Field(
        default="agentic",
        description="Parsing operational mode: 'fast' or 'agentic'."
    )


class Settings(BaseSettings):
    """
    Main application settings manager. 
    Loads environment variables from system environment or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="ignore",
    )

    env: str = Field(
        default="production", 
        description="Deployment environment stage (development/production/testing)."
    )
    
    # Nested Settings
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    parser: ParserSettings = Field(default_factory=ParserSettings)


# Safe Lazy Initialization Pattern
_settings_instance: Optional[Settings] = None

def get_settings() -> Settings:
    """Returns application singleton settings with safe lazy instantiation."""
    global _settings_instance
    if _settings_instance is None:
        try:
            _settings_instance = Settings()
        except Exception as err:
            logger.error(f"Failed to load environment configuration settings: {err}")
            _settings_instance = Settings(_env_file=None)
    return _settings_instance

# Global Instance
settings = get_settings()