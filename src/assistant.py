from typing import Tuple, Optional

from fastapi import HTTPException

from logger import logger
from services.llm_service import LLMService
from services.context_service import ContextService
from services.web_search_service import WebSearchService
from services.vector_store_service import VectorStoreService
from services.classify_service import is_conversational, is_islamic_finance_related, is_web_search_required, is_web_search_required


context_service = ContextService(message_cap=2)
llm_service = LLMService()
web_search_service = WebSearchService()
vector_store_service = VectorStoreService()

class Assistant():
    def answer_to_query(self, chat_id: int, query: str) -> Tuple[str, list, list]:
        logger.info(f"Question processing running...")

        chat_history = context_service.get_summarized_chat_history(chat_id)

        if is_conversational(query).response:
            answer = llm_service.generate_conversational_response(query, chat_history)
            context_service.add_data_to_chat_history(chat_id, query, answer)
            return answer, [], []

        question_with_history = f"История чата: {chat_history}\nВопрос: {query}"

        try:
            if not is_islamic_finance_related(query, chat_history).response:
                answer = "Извините, я могу отвечать только на вопросы, связанные с исламским финансированием. " \
                        "Пожалуйста, переформулируйте ваш вопрос или задайте вопрос, относящийся к этой теме."
                logger.info(f"Question is out of topic")
                return answer, [], []

            if not is_web_search_required(question_with_history).response:
                retrieved_info = vector_store_service.search_relevant_docs(query, k=5, search_type="similarity")
                answer, source_text = llm_service.generate_response_from_db(question_with_history, retrieved_info)
                context_service.add_data_to_chat_history(chat_id, query, answer)
                return answer, source_text, []
            else:
                search_results = web_search_service.web_search(query)
                retrieved_info = web_search_service.sort_sources(query, search_results)
                answer, source_text, urls = llm_service.generate_response_from_web(question_with_history, retrieved_info)
                context_service.add_data_to_chat_history(chat_id, query, answer)
                return answer, source_text, urls

        except HTTPException:
            return "Произошла ошибка при обработке вашего вопроса. Пожалуйста, попробуйте еще раз позже.", [], []
