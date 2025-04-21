from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class QuestionCategory(str, Enum):
    STATIC = "static"
    WEB_SEARCH = "web_search"

class QuestionTopic(str, Enum):
    ISLAMIC_FINANCE = "islamic_finance"
    OTHER = "other"

class QueryRequest(BaseModel):
    chat_id: int
    question: str

class GreetingRequest(BaseModel):
    chat_id: int

class QueryResponse(BaseModel):
    answer: str
    source_text: Optional[list[str]] = None
    urls: Optional[list[str]] = None

class QuestionCategoryClassification(BaseModel):
    category: QuestionCategory = Field(description="Категория вопроса")

class QuestionTopicClassification(BaseModel):
    topic: QuestionTopic = Field(description=" Тема вопроса")