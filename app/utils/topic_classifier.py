from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from enum import Enum

from config import settings

class QuestionCategory(str, Enum):
    ISLAMIC_FINANCE = "islamic_finance"
    OTHER = "other"

class QuestionClassification(BaseModel):
    category: QuestionCategory = Field(description="Категория вопроса")

llm = ChatOpenAI(
    openai_api_key=settings.OPENROUTER_API_KEY,
    model_name="google/gemini-2.0-flash-lite-preview-02-05:free",
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)


prompt = PromptTemplate.from_template(
    "Классифицируй следующий вопрос как 'islamic_finance' или 'other'. "
    "'islamic_finance' — это вопросы, связанные с исламскими финансами, исламским банкингом или шариатскими принципами в финансах. "
    "'other' — это вопросы, не связанные с этими темами.\n\n"
    "Примеры:\n"
    "- Вопрос: Что такое сукук?\n  Категория: islamic_finance\n"
    "- Вопрос: Как работает мурабаха?\n  Категория: islamic_finance\n"
    "- Вопрос: Что такое процентная ставка?\n  Категория: other\n"
    "- Вопрос: Каковы принципы шариата в финансах?\n  Категория: islamic_finance\n"
    "- Вопрос: Что такое акции?\n  Категория: other\n\n"
    "Теперь классифицируй этот вопрос:\n"
    "Вопрос: {question}\n\n"
    "Ответ: категория"
)

chain = prompt | llm.with_structured_output(QuestionClassification)

def classify_question(question: str) -> QuestionClassification:
    return chain.invoke({"question": question})