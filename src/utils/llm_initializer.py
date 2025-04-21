from logger import logger
from pydantic import SecretStr
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_openai import ChatOpenAI


def get_mistral_llm(api_key: str, model_name: str, temperature: int = 0.8) -> ChatMistralAI:
    if api_key == "":
        logger.error("Mistral API key is not found")
        exit(1)

    try:
        llm = ChatMistralAI(
            api_key=SecretStr(api_key),
            model_name=model_name,
            temperature=temperature,
        )
        return llm
    except Exception as e:
        logger.error(f"Error with ChatMistralAI initialization: {e}")
        exit(1)

def get_openaicomp_model(api_key: str, model_name: str, temperature: float = 0.7) -> ChatOpenAI:
    if api_key == "":
        logger.error("OpenAI compatible API key is not found")
        exit(1)

    try:
        llm = ChatOpenAI(
            openai_api_key=SecretStr(api_key),
            model_name=model_name,
            base_url="https://router.requesty.ai/v1",
            temperature=temperature
        )
        return llm
    except Exception as e:
        logger.error(f"Error with ChatOpenAI initialization: {e}")
        exit(1)

def get_embedding_model(api_key: str) -> MistralAIEmbeddings:
    if api_key == "":
        logger.error("Mistral API key is not found")
        exit(1)

    try:
        embeddings = MistralAIEmbeddings(api_key=SecretStr(api_key))
        return embeddings
    except Exception as e:
        logger.error(f"Error with MistralAIEmbeddings initialization: {e}")
        exit(1)