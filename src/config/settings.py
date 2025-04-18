import os
from pydantic_settings import BaseSettings

class MistralSettings(BaseSettings):
    api_key: str = os.getenv("MISTRAL_API_KEY", "")
    small_model: str = "mistral-small-latest"
    medium_model: str = "mistral-medium-latest"
    large_model: str = "mistral-large-latest"
    embedding_model: str = "mistral-embed"

class OpenRouterSettings(BaseSettings):
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    model: str = "google/gemini-2.0-flash-exp:free"

class RedisSettings(BaseSettings):
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379")

class TavilySettings(BaseSettings):
    api_key: str = os.getenv("TAVILY_API_KEY", "")

class Settings(BaseSettings):
    mistral: MistralSettings = MistralSettings()
    openrouter: OpenRouterSettings = OpenRouterSettings()
    redis: RedisSettings = RedisSettings()
    tavily: TavilySettings = TavilySettings()

settings = Settings()