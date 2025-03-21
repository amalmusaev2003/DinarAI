from fastapi import FastAPI

from utils.topic_classifier import classify_question
from services.llm_service import LLMService
from services.search_service import SearchService
from services.sort_source_service import SortSourceService
from services.context_service import ContextService
from schemas import QueryRequest, QueryResponse
from logger import logger

app = FastAPI()

search_service = SearchService()
sort_source_service = SortSourceService()
context_service = ContextService()
assistant = LLMService()

@app.get("/")
def root():
    return {"message": "Добро пожаловать в DinarAI API!"}

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    logger.info(f"Обработка запроса от chat_id {request.chat_id}: {request.question}")

    if classify_question(request.question).category == "other":
        answer = "Извините, я могу отвечать только на вопросы, связанные с исламским финансированием. " \
                 "Пожалуйста, переформулируйте ваш вопрос или задайте вопрос, относящийся к этой теме."
        return QueryResponse(answer=answer, source_text=[], sources=[])

    search_results = search_service.web_search(request.question)
    sorted_results = sort_source_service.sort_sources(request.question, search_results)
    chat_history = context_service.get_summarized_chat_history(request.chat_id)

    answer, source_text, sources = assistant.generate_response(request.question, sorted_results, chat_history)

    context_service.add_data_to_chat_history(request.chat_id, request.question, answer)
    return QueryResponse(answer=answer, source_text=source_text, sources=sources)