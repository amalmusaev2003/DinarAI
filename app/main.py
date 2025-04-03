from fastapi import FastAPI

from assistant import Assistant
from schemas.query import QueryRequest, QueryResponse

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Добро пожаловать в DinarAI API!"}

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    assistant = Assistant()
    answer, source_text, urls = assistant.answer_to_query(request.chat_id, request.question)
    return QueryResponse(answer=answer, source_text=source_text, urls=urls)