from typing import List, Tuple
from langchain.prompts import PromptTemplate

from logger import logger
from config import LLM_CONFIG


class LLMService:
    def generate_response(self, question: str, search_results: List[str], chat_history: str) -> Tuple[str, List[str]]:
        source_texts = [
            f"Источник {i+1} ({result['url']}):\n{result['content']}"
            for i, result in enumerate(search_results)
        ]

        context = "\n\n".join(source_texts)

        sources = [result['url'] for result in search_results]

        prompt = PromptTemplate(
            input_variables=["search_results", "chat_history", "question"],
            template="""Вы — эксперт по исламскому финансированию.
            На основе результатов поиска: {search_results} и истории чата: {chat_history} предоставьте актуальную информацию,
            которая удовлетворит запросу пользователя: {question}.
            """
        )

        chain = prompt | LLM_CONFIG["pro"]
        answer = chain.invoke({"search_results": context, "chat_history": chat_history, "question": question})

        final_answer = answer.content if hasattr(answer, 'content') else str(answer)
        logger.info(f"Ответ сформирован для вопроса: {question}")

        return final_answer, source_texts, sources
