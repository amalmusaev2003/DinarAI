from logger import logger
from pydantic import SecretStr
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_openai import ChatOpenAI


def get_mistral_llm(api_key: str, model_name: str) -> ChatMistralAI:
    if api_key == "":
        logger.error("API ключ Mistral не найден")
        exit(1)
    return ChatMistralAI(api_key=SecretStr(api_key), model_name=model_name)

def get_openrouter_llm(api_key: str, model_name: str, temperature: float = 0.7) -> ChatOpenAI:
    if api_key == "":
        logger.error("API OpenRouter не найден")
        exit(1)
    return ChatOpenAI(
        openai_api_key=SecretStr(api_key),
        model_name=model_name,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature
    )

def get_embedding_model(api_key: str) -> MistralAIEmbeddings:
    if api_key == "":
        logger.error("API ключ Mistral не найден")
        exit(1)
    return MistralAIEmbeddings(api_key=SecretStr(api_key))