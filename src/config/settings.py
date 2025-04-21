import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


load_dotenv()


class MistralSettings(BaseSettings):
    api_key: str = os.getenv("MISTRAL_API_KEY", "")
    small_model: str = "mistral-small-latest"
    medium_model: str = "mistral-medium-latest"
    large_model: str = "mistral-large-latest"
    embedding_model: str = "mistral-embed"

class OpenAICompSettings(BaseSettings):
    api_key: str = os.getenv("OPENAICOMP_API_KEY", "")
    model: str = "google/gemini-2.0-flash-exp"

class RedisSettings(BaseSettings):
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379")

class TavilySettings(BaseSettings):
    api_key: str = os.getenv("TAVILY_API_KEY", "")

class Settings(BaseSettings):
    mistral: MistralSettings = MistralSettings()
    openai_comp: OpenAICompSettings = OpenAICompSettings()
    redis: RedisSettings = RedisSettings()
    tavily: TavilySettings = TavilySettings()

settings = Settings()