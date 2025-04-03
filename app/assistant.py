from typing import Tuple, Optional

from logger import logger
from services.llm_service import LLMService
from services.context_service import ContextService
from services.web_search_service import WebSearchService
from services.vector_store_service import VectorStoreService
from utils.classifier import classify_question_category, classify_question_topic
from exceptions import ContextServiceError, DinarAIException, VectorStoreError, LLMServiceError, WebSearchServiceError


context_service = ContextService()
llm_service = LLMService()
web_search_service = WebSearchService()
vector_store_service = VectorStoreService()

class Assistant():
    def answer_to_query(self, chat_id: int, question: str) -> Tuple[str, list, Optional[list]]:
        try:
            logger.info(f"Question processing running...")

            logger.info(f"Question topic classification running...")
            if classify_question_topic(question).topic == "other":
                answer = "Извините, я могу отвечать только на вопросы, связанные с исламским финансированием. " \
                        "Пожалуйста, переформулируйте ваш вопрос или задайте вопрос, относящийся к этой теме."
                logger.info(f"Question is out of topic")
                return answer, [], None

            try:
                chat_history = context_service.get_summarized_chat_history(chat_id)
            except Exception as e:
                logger.error(f"Context service error: {e}")
                raise ContextServiceError()
            
            if classify_question_category(question).category == "static":
                try:
                    retrieved_info = vector_store_service.search_relevant_docs(question, k=5, search_type="similarity")
                except Exception as e:
                    logger.error(f"Error while vector store retrieval: {e}")
                    raise VectorStoreError()
                try:
                    answer, source_text = llm_service.generate_response_from_db(question, retrieved_info, chat_history)
                    return answer, source_text, None
                except Exception as e:
                    logger.error(f"Answer generation error: {e}")
                    raise LLMServiceError()
            else:
                try:
                    search_results = web_search_service.web_search(question)
                    retrieved_info = web_search_service.sort_sources(question, search_results)
                except Exception as e:
                    logger.error(f"Web-search error: {e}")
                    raise WebSearchServiceError()

                try:
                    answer, source_text, urls = llm_service.generate_response_from_web(question, retrieved_info, chat_history)
                    return answer, source_text, urls
                except Exception as e:
                    logger.error(f"Answer generation error: {e}")
                    raise LLMServiceError()

        except DinarAIException as e:
            logger.error(f"Question pocessing error: {e}")
            return "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз.", [], None
