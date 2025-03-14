from langchain_mistralai import ChatMistralAI
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
#from langchain_openai import ChatOpenAI # Для использования OpenAI совместимых API

load_dotenv()

class Settings(BaseSettings):
    MISTRAL_API_KEY: str = None
    OPENROUTER_API_KEY: str = None
    TAVILY_API_KEY: str
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

settings = Settings()

# Конфигурация LLM для разных режимов
LLM_CONFIG = {
    "classic": ChatMistralAI(api_key=settings.MISTRAL_API_KEY, model="mistral-medium-latest"),
    "pro": ChatMistralAI(api_key=settings.MISTRAL_API_KEY, model="mistral-large-latest")
}