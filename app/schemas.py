from pydantic import BaseModel
from typing import Optional, List, Union, Dict

class QueryRequest(BaseModel):
    chat_id: int
    question: str

class QueryResponse(BaseModel):
    answer: Union[str, Dict]
    sources: Optional[List[str]] = None