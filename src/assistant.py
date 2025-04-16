from typing import Tuple, Optional

from fastapi import HTTPException

from logger import logger
from services.llm_service import LLMService
from services.context_service import ContextService
from services.web_search_service import WebSearchService
from services.vector_store_service import VectorStoreService
from utils.classifier import classify_question_category, classify_question_topic


context_service = ContextService()
llm_service = LLMService()
web_search_service = WebSearchService()
vector_store_service = VectorStoreService()

class Assistant():
    def answer_to_query(self, chat_id: int, question: str) -> Tuple[str, list, Optional[list]]:
        logger.info(f"Question processing running...")

        chat_history = context_service.get_summarized_chat_history(chat_id, messages=2)
        
        question_with_history = f"История чата: {chat_history}\nВопрос: {question}"
        logger.info(f"Question with chat history: {question_with_history}")

        try:
            if classify_question_topic(question, chat_history).topic == "other":
                answer = "Извините, я могу отвечать только на вопросы, связанные с исламским финансированием. " \
                        "Пожалуйста, переформулируйте ваш вопрос или задайте вопрос, относящийся к этой теме."
                logger.info(f"Question is out of topic")
                return answer, [], None

            if classify_question_category(question_with_history).category == "static":
                retrieved_info = vector_store_service.search_relevant_docs(question, k=5, search_type="similarity")
                answer, source_text = llm_service.generate_response_from_db(question_with_history, retrieved_info)
                context_service.add_data_to_chat_history(chat_id, question, answer)
                return answer, source_text, None
            else:
                search_results = web_search_service.web_search(question)
                retrieved_info = web_search_service.sort_sources(question, search_results)
                answer, source_text, urls = llm_service.generate_response_from_web(question_with_history, retrieved_info)
                context_service.add_data_to_chat_history(chat_id, question, answer)
                return answer, source_text, urls
        except HTTPException:
            return "Произошла ошибка при обработке вашего вопроса. Пожалуйста, попробуйте еще раз позже.", [], None
