from fastapi import HTTPException
from langchain.prompts import PromptTemplate

from logger import logger
from config.settings import settings
from utils.llm_initializer import get_openaicomp_model
from schemas.query import QuestionTopicClassification, QuestionCategoryClassification


llm_settings = settings.openai_comp
llm = get_openaicomp_model(llm_settings.api_key, llm_settings.model, temperature=0)

def classify_question_topic(question: str, context: str) -> QuestionTopicClassification:
    prompt_template = PromptTemplate.from_template(
        "Классифицируй следующий вопрос пользователя как 'islamic_finance' или 'other'. "
        "'islamic_finance' — это вопросы, связанные с исламскими финансами, исламским банкингом или шариатскими принципами в финансах. "
        "'other' — это вопросы, не связанные с этими темами.\n\n"
        "Вопрос: {question}\n\n"
        "Анализируй не только вопрос, но и контекст предыдущего разговора для более точной классификации. "
        "Если вопрос кажется не относящимся к теме, но в контексте разговора он связан с исламскими финансами, "
        "классифицируй его как 'islamic_finance'.\n\n"
        "Если вопрос не относится к исламским финансам и не имеет отношения к контексту, то классифицируй его как 'other'.\n\n"
        "Примеры:\n"
        "- Вопрос: Что такое сукук?\n  Тема: islamic_finance\n"
        "- Вопрос: Как работает мурабаха?\n  Тема: islamic_finance\n"
        "- Вопрос: Что такое процентная ставка?\n  Тема: other\n"
        "- Вопрос: Каковы принципы шариата в финансах?\n  Тема: islamic_finance\n"
        "- Вопрос: Что такое акции?\n  Тема: other\n\n"
        "Теперь классифицируй этот запрос с контекстом:\n"
        "{context}\n\n"
        "Ответ: тема"
    )

    llm_with_structured_output = llm.with_structured_output(QuestionTopicClassification)
    prompt = prompt_template.invoke({"question": question, "context": context})

    try:
        answer = llm_with_structured_output.invoke(prompt)
        logger.info(f"Topic: {answer}")
        return answer
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to classify question topic (islamic-finance/other)")


def classify_question_category(question: str) -> QuestionCategoryClassification:
    prompt_template = PromptTemplate.from_template(
        "Классифицируй следующий вопрос пользователя как 'static' или 'web_search'. "
        "'static' означает, что ответ можно найти в векторной базе данных с редко обновляемыми данными. "
        "'web_search' означает, что для ответа требуется актуальная информация из интернета.\n\n"
        "Вопрос: {question}\n\n"
        "Если вопрос требует актуальных данных или относится к текущим событиям, выбирай 'web_search'. "
        "Если вопрос касается общих принципов, концепций или исторически установленных практик, выбирай 'static'.\n\n"
        "Ответ: категория"
    )

    llm_with_structured_output = llm.with_structured_output(QuestionCategoryClassification)
    prompt = prompt_template.invoke({"question": question})

    try:
        answer = llm_with_structured_output.invoke(prompt)
        logger.info(f"Question category: {answer}")
        return answer
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to classify question category (static/web-search)")