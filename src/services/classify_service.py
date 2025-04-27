from fastapi import HTTPException
from langchain.prompts import PromptTemplate

from logger import logger
from config.settings import settings
from utils.llm_initializer import get_openaicomp_model
from schemas.query import QueryClassification


llm_settings = settings.openai_comp
llm = get_openaicomp_model(llm_settings.api_key, llm_settings.model, temperature=0)


def is_conversational(message: str) -> QueryClassification:
    prompt_template = PromptTemplate.from_template(
        "Определи, является ли следующее сообщение пользователя разговорным или нет. "
        "Разговорное сообщение - это приветствие, благодарность, вопрос о боте, или другая фраза, "
        "не требующая специальных знаний для ответа. "
        "Ответь только 'True' если сообщение разговорное или 'False' если нет.\n\n"
        "Сообщение: {message}\n\n"
        "Примеры разговорных сообщений:\n"
        "- Привет, как дела?\n"
        "- Как тебя зовут?\n"
        "- Спасибо за информацию\n"
        "- Что ты умеешь делать?\n"
        "- Ты бот или человек?\n\n"
        "- Кто тебя создал?\n"
        "Примеры не разговорных сообщений:\n"
        "- Что такое мурабаха?\n"
        "- Какие банки предлагают исламское финансирование?\n"
        "- Объясни принципы исламского банкинга\n"
        "- Как работает сукук?\n\n"
        "Ответь только 'True' или 'False'."
    )

    llm_with_structured_output = llm.with_structured_output(QueryClassification)
    prompt = prompt_template.invoke({"message": message})

    try:
        response = llm_with_structured_output.invoke(prompt)
        logger.info(f"Is conversational: {response}")
        return response
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to classify if question is conversational or not.")

def is_islamic_finance_related(message: str, context: str) -> QueryClassification:
    prompt_template = PromptTemplate.from_template(
        "Классификация: связан ли вопрос с исламскими финансами (True/False)?\n\n"
        "Исламские финансы включают: партнерское финансирование, исламский банкинг, шариатские принципы, "
        "термины: сукук, мурабаха, иджара, такафул.\n\n"
        "Правила классификации:\n"
        "1. Если вопрос напрямую связан с исламскими финансами → True\n"
        "2. Если вопрос косвенно связан с исламскими финансами через контекст → True\n"
        "3. Если вопрос явно не относится к исламским финансам, даже если контекст о них → False\n\n"
        "Примеры True:\n"
        "- Вопрос: 'Что такое сукук?' → True (прямая связь)\n"
        "- Контекст: 'Иджара - это исламский финансовый инструмент...', Вопрос: 'Чем она отличается от классического лизинга?' → True (косвенная связь)\n\n"
        "Примеры False:\n"
        "- Вопрос: 'Что такое квантовая механика?' → False (нет связи)\n"
        "- Контекст о исламских финансах, Вопрос: 'Как испечь торт?' → False (явно не относится к теме)\n"
        "- Контекст о исламских финансах, Вопрос: 'Я лучший баскетболист в мире' → False (явно не относится к теме)\n\n"
        "Вопрос: {message}\n"
        "Контекст: {context}\n\n"
        "Ответ (только True/False):"
    )

    llm_with_structured_output = llm.with_structured_output(QueryClassification)
    prompt = prompt_template.invoke({"message": message, "context": context})

    try:
        response = llm_with_structured_output.invoke(prompt)
        logger.info(f"Is query related to islamic banking: {response}")
        return response
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to classify question topic (islamic-finance/other)")


def is_web_search_required(message: str) -> QueryClassification:
    prompt_template = PromptTemplate.from_template(
        "Определи, требуется ли веб-поиск для ответа на следующий вопрос пользователя. "
        "Веб-поиск требуется, если для ответа нужна актуальная информация из интернета. "
        "Веб-поиск не требуется, если ответ можно найти в векторной базе данных с редко обновляемыми данными. "
        "Ответь только 'True' если требуется веб-поиск или 'False' если нет.\n\n"
        "Вопрос: {message}\n\n"
        "Если вопрос требует актуальных данных или относится к текущим событиям, ответь 'True'. "
        "Если вопрос касается общих принципов, концепций или исторически установленных практик, ответь 'False'.\n\n"
        "Ответь только 'True' или 'False'."
    )

    llm_with_structured_output = llm.with_structured_output(QueryClassification)
    prompt = prompt_template.invoke({"message": message})

    try:
        response = llm_with_structured_output.invoke(prompt)
        logger.info(f"Is web-search needed to answer the query: {response}")
        return response
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to classify question category (static/web-search)")