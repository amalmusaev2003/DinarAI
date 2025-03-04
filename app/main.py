from fastapi import FastAPI

from assistant import Assistant
from schemas import QueryRequest, QueryResponse

app = FastAPI()

assistant = Assistant()


@app.get("/")
def root():
    return {"message": "Добро пожаловать в DinarAI API!"}

@app.post("/learn", response_model=QueryResponse)
async def learn_topic(request: QueryRequest):
    answer = assistant.get_response(request.question)
    return QueryResponse(answer=answer)

@app.post("/latest", response_model=QueryResponse)
async def get_latest(request: QueryRequest):
    answer, sources = assistant.get_response(request.question, "latest")
    return QueryResponse(answer=answer, sources=sources)