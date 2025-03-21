from typing import List
from langchain_mistralai import MistralAIEmbeddings

from logger import logger
from config import settings


class SortSourceService:
    def __init__(self):
        self.embedding_model = MistralAIEmbeddings(model="mistral-embed", api_key=settings.MISTRAL_API_KEY)

    def sort_sources(self, query: str, search_results: List[dict]):
        try:
            logger.info(f"Сортировка результатов поиска для запроса '{query}' началась.")
            relevant_docs = []

            for res in search_results:
                relevance_score = res.get("relevance_score", 0)
                if relevance_score > 0.3:
                    relevant_docs.append(res)

            return sorted(relevant_docs, key=lambda x: x["relevance_score"], reverse=True)
        
        except Exception as e:
            logger.error(f"Ошибка при сортировке результатов поиска: {e}")
            return []