import os
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
#from langchain_openai import ChatOpenAI

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("Задайте MISTRAL_API_KEY в переменной окружения")
if not OPENROUTER_API_KEY:
    raise ValueError("Задайте OPENROUTER_API_KEY в переменной окружения")

EMBEDDING_MODEL = MistralAIEmbeddings(api_key=MISTRAL_API_KEY, model="mistral-embed",)
# Альтернатива: EMBEDDING_MODEL = SentenceTransformersEmbeddings(model_name="all-MiniLM-L6-v2")

# Конфигурация LLM для разных режимов
LLM_CONFIG = {
    "learn": ChatMistralAI(api_key=MISTRAL_API_KEY, model="mistral-medium-latest"),
    "latest": ChatMistralAI(api_key=MISTRAL_API_KEY, model="mistral-medium-latest")
}

CHROMA_DB_PATH = "../chroma_db"