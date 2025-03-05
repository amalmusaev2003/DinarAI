from fastapi import FastAPI

from assistant import Assistant
from schemas import QueryRequest, QueryResponse

app = FastAPI()

assistant = Assistant()

@app.get("/")
def root():
    return {"message": "Добро пожаловать в DinarAI API!"}

@app.post("/ask", response_model=QueryResponse)
async def get_latest(request: QueryRequest):
    answer, sources = assistant.get_response(request.question, request.chat_id)
    return QueryResponse(answer=answer, sources=sources)