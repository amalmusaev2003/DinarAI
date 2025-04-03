from langchain.prompts import PromptTemplate

from logger import logger
from utils.llm_initializer import get_openrouter_llm
from config.settings import settings
from schemas.query import QuestionTopicClassification, QuestionCategoryClassification, QuestionTopic, QuestionCategory
from exceptions import TopicClassificationError, CategoryClassificationError


llm_settings = settings.openrouter
llm = get_openrouter_llm(llm_settings.api_key, llm_settings.regular_model)

def classify_question_topic(question: str) -> QuestionTopicClassification:
    prompt = PromptTemplate.from_template(
        "Классифицируй следующий вопрос как 'islamic_finance' или 'other'. "
        "'islamic_finance' — это вопросы, связанные с исламскими финансами, исламским банкингом или шариатскими принципами в финансах. "
        "'other' — это вопросы, не связанные с этими темами.\n\n"
        "Примеры:\n"
        "- Вопрос: Что такое сукук?\n  Тема: islamic_finance\n"
        "- Вопрос: Как работает мурабаха?\n  Тема: islamic_finance\n"
        "- Вопрос: Что такое процентная ставка?\n  Тема: other\n"
        "- Вопрос: Каковы принципы шариата в финансах?\n  Тема: islamic_finance\n"
        "- Вопрос: Что такое акции?\n  Тема: other\n\n"
        "Теперь классифицируй этот вопрос:\n"
        "Вопрос: {question}\n\n"
        "Ответ: тема"
    )

    chain = prompt | llm.with_structured_output(QuestionTopicClassification)

    answer = chain.invoke({"question": question})
    if isinstance(answer, QuestionTopicClassification):
        if answer.topic in {QuestionTopic.ISLAMIC_FINANCE, QuestionTopic.OTHER}:
            return answer
        else:
            raise TopicClassificationError
    else:
        raise TopicClassificationError


def classify_question_category(question: str) -> QuestionCategoryClassification:
    prompt = PromptTemplate.from_template(
        "Классифицируй следующий вопрос как 'static' или 'web_search'. "
        "'static' означает, что ответ можно найти в векторной базе данных с редко обновляемыми данными, "
        "а 'web_search' означает, что для ответа требуется актуальная информация из интернета.\n\n"
        "Вопрос: {question}\n\n"
        "Ответ: категория"
    )

    chain = prompt | llm.with_structured_output(QuestionCategoryClassification)

    answer = chain.invoke({"question": question})
    if isinstance(answer, QuestionCategoryClassification):
        if answer.category in {QuestionCategory.STATIC, QuestionCategory.WEB_SEARCH}:
            return answer
        else:
            raise CategoryClassificationError
    else:
        raise CategoryClassificationError
