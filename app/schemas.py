from pydantic import BaseModel
from typing import Optional, List, Union, Dict

class QueryRequest(BaseModel):
    user_id: str
    question: str

class QueryResponse(BaseModel):
    answer: Union[str, Dict]
    sources: Optional[List[str]] = None