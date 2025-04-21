import json
import redis
from langchain.prompts import PromptTemplate

from logger import logger
from config.settings import settings
from utils.llm_initializer import get_openaicomp_model


llm_settings = settings.openai_comp
llm = get_openaicomp_model(llm_settings.api_key, llm_settings.model)

class ContextService:
    def __init__(self, message_cap: int):
        try:
            redis_settings = settings.redis
            self.redis_client = redis.from_url(
                redis_settings.url,
                decode_responses=True
            )
            self.message_cap = message_cap
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            exit(1)

    def _summarize_chat_history(self, chat_history: list) -> str:
        if not chat_history:
            return "Диалог пуст."

        history_summary_prompt = PromptTemplate(
            input_variables=["chat_history"],
            template="""Суммаризируй следующий диалог в краткое резюме (до 100 слов) на русском языке:
            {chat_history}
            """
        )
        chat_text = "\n".join([f"Q: {q} A: {a}" for q, a in chat_history])

        chain = history_summary_prompt | llm
        summary = chain.invoke({"chat_history": chat_text})
        return str(summary.content) if hasattr(summary, 'content') else str(summary)

    def get_chat_history(self, chat_id: int) -> list:
        context_history = self.redis_client.get(str(chat_id))
        if context_history:
            context_history = json.loads(str(context_history))
        else:
            context_history = []
        return context_history

    def get_summarized_chat_history(self, chat_id: int) -> str:
        context_history = self.get_chat_history(chat_id)
        if not context_history:
            return "Диалог пуст."
        
        chat_summary = self._summarize_chat_history(context_history)
        return chat_summary

    def add_data_to_chat_history(self,  chat_id: int, question: str, answer: str) -> None:
        context_history = self.get_chat_history(chat_id)
        context_history.append((question, answer))
        self.redis_client.set(str(chat_id), json.dumps(context_history[-self.message_cap:]))
        logger.info(f"Chat history for chat_id: {chat_id}: \n {context_history}")
