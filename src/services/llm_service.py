from typing import Tuple
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from logger import logger
from config.settings import settings
from utils.llm_initializer import get_openaicomp_model


llm_settings = settings.openai_comp
llm = get_openaicomp_model(llm_settings.api_key, llm_settings.model)

class LLMService:
    def __init__(self):
        self.system_prompt = PromptTemplate(
            input_variables=["search_results", "question"],
            template="""Ты — эксперт по исламскому финансированию.
            На основе результатов поиска: {search_results} предоставь актуальную информацию,
            которая удовлетворит запросу пользователя: {question}.
            Правила:
            - Отвечай только на запрос пользователя.
            - Используй информацию из предоставленных результатов поиска.
            - Если информация отсутствует, ответь, что не знаешь.
            - Ответь кратко и по существу.
            """
        )

        self.conversational_prompt = PromptTemplate(
            input_variables=["question", "chat_history"],
            template="""Ты — Динар, бот-помощник по исламскому финансированию.
            
            Ответь на сообщение пользователя: {question}

            История чата: {chat_history}

            Правила:
            1. Представляйся как Динар, бот-помощник по исламскому финансированию.
            2. Отвечай на приветствия, благодарности и вопросы о себе вежливо и кратко.
            3. Если пользователь спрашивает о твоих возможностях, объясни, что ты специализируешься на вопросах исламского финансирования.
            4. Не отвечай на вопросы, не связанные с исламским финансированием, кроме общих разговорных фраз.
            5. Будь дружелюбным, но профессиональным.
            6. Если тебя спросят о том, кто тебя создал, ответь, что твой создатель: Мусаев Амаль. С ним можно связаться в телеграмм: https://t.me/ar_musaev.

            Ответь одним абзацем, кратко и по существу.
            """
        )

    def generate_conversational_response(self, question: str, chat_history: str) -> str:
        chain = self.conversational_prompt | llm
        answer = chain.invoke({"question": question, "chat_history": chat_history})
        final_answer = str(answer.content) if hasattr(answer, 'content') else str(answer)
        logger.info(f"Answer is prepared for: \"{question}\"")

        return final_answer

    def generate_response_from_web(self, question: str, search_results: list[dict]) -> Tuple[str, list[str], list[str]]:
        source_text = [
            f"Источник {i+1} ({result['url']}):\n{result['content']}"
            for i, result in enumerate(search_results)
        ]

        context = "\n\n".join(source_text)

        sources = [result['url'] for result in search_results]

        chain = self.system_prompt | llm
        answer = chain.invoke({"search_results": context, "question": question})

        final_answer = str(answer.content) if hasattr(answer, 'content') else str(answer)
        logger.info(f"Answer is prepared for: \"{question}\"")

        return final_answer, source_text, sources
    
    def generate_response_from_db(self, question: str, search_results: list[Document]) -> Tuple[str, list[str]]:
        source_text = [
            f"{result.page_content}\n"
            for result in search_results
        ]

        context = "\n\n".join(source_text)
        pages = [result.metadata['page_label'] for result in search_results]
        logger.info(f"Pages included to answer: {pages}")
        
        chain = self.system_prompt | llm
        answer = chain.invoke({"search_results": context, "question": question})
        final_answer = str(answer.content) if hasattr(answer, 'content') else str(answer)
        logger.info(f"Answer is prepared for: \"{question}\"")

        return final_answer, source_text
