import json
import redis
from typing import Union, List
from langchain.prompts import PromptTemplate

from web_service import search_web
from logging_config import logger
from config import settings, LLM_CONFIG

class Assistant:
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            logger.info("Подключение к Redis успешно установлено")
        except Exception as e:
            logger.error(f"Ошибка подключения к Redis: {e}")
            raise

        self.main_template = PromptTemplate(
                input_variables=["search_results", "chat_history", "question"],
                template="""Вы — эксперт по исламскому финансированию. 
                На основе результатов поиска: {search_results} и истории чата: {chat_history} предоставьте актуальную информацию,
                которая удовлетворит запросу пользователя: {question}
                """
            )

        self.history_summary_prompt = PromptTemplate(
            input_variables=["chat_history"],
            template="""Суммаризируй следующий диалог в краткое резюме (до 100 слов) на русском языке:
            {chat_history}
            """
        )

    def _summarize_chat_history(self, chat_history: List[tuple]) -> str:
        if not chat_history:
            return "Диалог пуст."
        chat_text = "\n".join([f"Q: {q} A: {a}" for q, a in chat_history])
        try:
            chain = self.history_summary_prompt | LLM_CONFIG["classic"]
            summary = chain.invoke({"chat_history": chat_text})
            return summary.content if hasattr(summary, 'content') else str(summary)
        except Exception as e:
            logger.error(f"Ошибка при суммаризации истории чата: {e}")
            return "Не удалось создать резюме диалога."

    def get_response(self, question: str, chat_id: int, request_type: str = "learn") -> Union[str, List[str]]:
        logger.info(f"Обработка запроса от chat_id {chat_id}: {question}")

        context_history = self.redis_client.get(chat_id)
        if context_history:
            context_history = json.loads(context_history)
        else:
            context_history = []
        chat_summary = self._summarize_chat_history(context_history[-5:])

        summary, sources = search_web(question, llm=LLM_CONFIG["classic"])
        prompt = self.main_template
        chain = prompt | LLM_CONFIG["pro"]
        answer = chain.invoke({"chat_history": chat_summary, "question": question, "search_results": summary})
        final_answer = answer.content if hasattr(answer, 'content') else str(answer)
        logger.info(f"Ответ сформирован для вопроса: {question}")
        context_history.append((question, final_answer))
        self.redis_client.setex(chat_id, 3600, json.dumps(context_history[-10:]))
        return final_answer, sources