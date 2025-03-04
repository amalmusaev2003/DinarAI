from typing import Union, List
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma

from web_service import search_web
from logging_config import logger
from config import LLM_CONFIG, EMBEDDING_MODEL, CHROMA_DB_PATH

class Assistant:
    def __init__(self):
        self.vector_store = Chroma(
            embedding_function=EMBEDDING_MODEL,
            persist_directory=CHROMA_DB_PATH
        )
        self.templates = {
            "learn": PromptTemplate(
                input_variables=["question", "context"],
                template="""Вы — эксперт по исламскому финансированию.
                Используя следующий контекст: {context}, объясните подробно и с примерами на русском языке: {question}
                """
            ),
            "latest": PromptTemplate(
                input_variables=["search_results", "question"],
                template="""Вы — эксперт по исламскому финансированию. 
                На основе результатов поиска: {search_results} предоставьте актуальную информацию о состоянии исламского финансирования в России, 
                включая последние новости и доступные услуги (например, ипотека, мурабаха, сукук), по теме: {question}
                """
            )
        }

    def get_response(self, question: str, request_type: str = "learn") -> Union[str, List[str]]:
        logger.info(f"Обработка запроса: {question} в режиме {request_type}")

        if request_type == "latest":
            summary, sources = search_web(question, llm=LLM_CONFIG["latest"])
            logger.info(f"Результат из веб-поиска: {summary}")

            logger.info(f"Ответ сформирован для вопроса: {question}")
            return summary, sources

        prompt = self.templates.get(request_type, self.templates["learn"])
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
        )
        relevant_docs = retriever.invoke(question)

        if not relevant_docs:
            logger.warning(f"Не найдено релевантных документов для запроса: {question}")
            return "В базе данных нет информации по этому запросу."
        context = "\n".join([doc.page_content for doc in relevant_docs])
        logger.debug(f"Найден контекст: {context[:200]}...")
        chain = prompt | LLM_CONFIG["learn"]
        response = chain.invoke({"question": question, "context": context})
        logger.info("Ответ сформирован для вопроса: {question}")
        return response.content if hasattr(response, 'content') else str(response)