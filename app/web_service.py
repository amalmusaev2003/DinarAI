from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Tuple

from logging_config import logger


def search_web(query: str, llm: BaseChatModel) -> Tuple[str, List[str]]:
    try:
        logger.info(f"Начало веб-поиска по запросу: {query}")
        refine_prompt_template = """
        Ты — эксперт по оптимизации поисковых запросов для веб-поиска. На основе исходного вопроса пользователя:
        "{query}"
        Составь уточненный поисковый запрос, который:
        - Сфокусирован на ключевых аспектах вопроса.
        - Добавляет релевантные ключевые слова (например, "исламский банкинг", "услуги", "Россия", "2025").
        - Предпочитает русскоязычные результаты (добавь "site:.ru", если применимо).
        Верни только сам запрос, без пояснений.
        """
        refine_prompt = PromptTemplate(template=refine_prompt_template, input_variables=["query"])
        refine_chain = refine_prompt | llm

        logger.debug("Генерация уточненного запроса...")
        refined_query = refine_chain.invoke({"query": query}).content.strip()
        logger.info(f"Уточненный запрос: {refined_query}")

        search = DuckDuckGoSearchResults(output_format="list", num_results=10)
        logger.debug(f"Выполнение поиска через DuckDuckGo: {refined_query}")
        search_results = search.run(refined_query)
        logger.debug(f"Сырые результаты поиска: {search_results}")
        logger.debug(f"Полученные результаты поиска: {search_results[:200]}...")

        if not search_results:
            logger.warning("Поиск не дал результатов")
            return "Не удалось получить результаты поиска", []

        urls = [result['link'] for result in search_results]
        logger.debug(f"Извлеченные URL: {urls}")

        formatted_results = "\n\n".join([
            f"Источник: {result['title']}\n{result['snippet']}"
            for result in search_results
        ])

        prompt_template = """
        Ты — помощник, который кратко суммирует информацию. На основе следующих результатов поиска:
        {search_results}
        Подробно ответь на вопрос: {query}
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["search_results", "query"])
        chain = prompt | llm

        logger.debug("Начало суммирования результатов")
        response = chain.invoke({"search_results": formatted_results, "query": query})

        summary = response.content if hasattr(response, 'content') else str(response)
        summary = summary.strip()

        logger.info("Поиск и суммаризация успешно завершены")
        return summary, urls

    except Exception as e:
        logger.error(f"Ошибка при выполнении веб-поиска: {e}", exc_info=True)
        return "Не удалось выполнить поиск из-за технической ошибки. Попробуйте позже.", []