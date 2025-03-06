import json
import redis
from langchain.prompts import PromptTemplate

from config import settings, LLM_CONFIG
from logger import logger


class ContextService:
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True
            )
        except Exception as e:
            logger.error(f"Ошибка при подключении к Redis: {e}")
            raise
    
    def _summarize_chat_history(self, chat_history: list[tuple]) -> str:
        if not chat_history:
            return "Диалог пуст."

        history_summary_prompt = PromptTemplate(
            input_variables=["chat_history"],
            template="""Суммаризируй следующий диалог в краткое резюме (до 100 слов) на русском языке:
            {chat_history}
            """
        )
        chat_text = "\n".join([f"Q: {q} A: {a}" for q, a in chat_history])
        try:
            chain = history_summary_prompt | LLM_CONFIG["classic"]
            summary = chain.invoke({"chat_history": chat_text})
            return summary.content if hasattr(summary, 'content') else str(summary)
        except Exception as e:
            logger.error(f"Ошибка при суммаризации истории чата: {e}")
            return "Не удалось создать резюме диалога."

    def get_chat_history(self, chat_id: int) -> list:
        context_history = self.redis_client.get(chat_id)
        if context_history:
            context_history = json.loads(context_history)
        else:
            context_history = []
        return context_history

    def get_summarized_chat_history(self, chat_id: int) -> str:
        context_history = self.get_chat_history(chat_id)
        if not context_history:
            return "Диалог пуст."
        
        chat_summary = self._summarize_chat_history(context_history[-5:])
        return chat_summary
    
    def add_data_to_chat_history(self,  chat_id: int, question: str, answer: str) -> None:
        context_history = self.get_chat_history(chat_id)
        context_history.append((question, answer))
        self.redis_client.setex(chat_id, 3600, json.dumps(context_history[-10:]))

