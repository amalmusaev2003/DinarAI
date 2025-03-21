from config import Settings
from tavily import TavilyClient

from logger import logger


settings = Settings()
tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)


class SearchService:
    def web_search(self, query: str):
        try:
            logger.info(f"Поиск в интернете по запросу {query} начался.")
            results = []
            response = tavily_client.search(query, max_results=10)
            search_results = response.get("results", [])

            for result in search_results:
                results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "relevance_score": result.get("score", 0),
                        "content": result.get("content", ""),
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Ошибка при поиске в интернете: {e}")