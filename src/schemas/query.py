from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    chat_id: int
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_text: Optional[list[str]] = None
    urls: Optional[list[str]] = None

class QueryClassification(BaseModel):
    response: bool = Field(description="Классификация сообщения")