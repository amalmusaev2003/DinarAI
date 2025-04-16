from langchain_chroma import Chroma

from logger import logger
from config.settings import settings
from utils.llm_initializer import get_embedding_model


llm_settings = settings.mistral
embedding_model = get_embedding_model(llm_settings.api_key)

class VectorStoreService:
    def __init__(self, collection_name: str="documents", vectordb_path: str="chroma"):
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=vectordb_path,
        )

    def search_relevant_docs(self, query: str, k: int, search_type: str) -> list:
        logger.info(f"Vector database retrieval running...")
        retriver = self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})
        relevant_docs = retriver.invoke(query)
        return relevant_docs
