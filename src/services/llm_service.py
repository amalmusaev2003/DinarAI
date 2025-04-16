from typing import Tuple
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from logger import logger
from config.settings import settings
from utils.llm_initializer import get_openrouter_llm


llm_settings = settings.openrouter
llm = get_openrouter_llm(llm_settings.api_key, "meta-llama/llama-4-scout:free")

class LLMService:
    def __init__(self):
        self.system_prompt = PromptTemplate(
            input_variables=["search_results", "question"],
            template="""Вы — эксперт по исламскому финансированию.
            На основе результатов поиска: {search_results} предоставьте актуальную информацию,
            которая удовлетворит запросу пользователя: {question}.
            """
        )

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
