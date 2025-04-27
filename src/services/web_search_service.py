from tavily import TavilyClient, UsageLimitExceededError

from logger import logger
from config.settings import settings


tavily_client = TavilyClient(api_key=settings.tavily.api_key)

class WebSearchService:
    def web_search(self, query: str) -> list[dict]:
        logger.info(f"Web-search on \"{query}\" running...")
        results = []

        domains_to_include = [
            "https://www.sberbank.ru/ru/person/islamic-banking"
        ]
        try:
            response = tavily_client.search(query, max_results=10)
            additional_response = tavily_client.search(query, include_domains=domains_to_include)
        except UsageLimitExceededError as e:
            logger.error(f"Usage limit exceeded. Please check your plan's usage limits or consider upgrading.")

        search_results = response.get("results", [])
        additional_results = additional_response.get("results", [])

        search_results.extend(additional_results)

        for result in search_results:
            print(result.get("url", ""))
            results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "relevance_score": result.get("score", 0),
                    "content": result.get("content", ""),
                }
            )

        return results

    def sort_sources(self, query: str, search_results: list[dict]):
        logger.info(f"Сортировка результатов поиска для запроса '{query}' началась.")
        relevant_docs = []

        for res in search_results:
            relevance_score = res.get("relevance_score", 0)
            if relevance_score > 0.3:
                relevant_docs.append(res)

        return sorted(relevant_docs, key=lambda x: x["relevance_score"], reverse=True)