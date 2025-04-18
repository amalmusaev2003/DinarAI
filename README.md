# DinarAI

DinarAI - это API-сервис, специализирующийся на обработке запросов по исламскому финансированию с использованием искусственного интеллекта.

## Описание

Проект представляет собой FastAPI приложение, которое обрабатывает пользовательские запросы, связанные с исламским финансированием. Система использует современные языковые модели для генерации ответов на основе проверенных источников информации.

## Основные компоненты

- **Question Validator** - проверяет, относится ли вопрос к теме исламского финансирования
- **Search Service** - выполняет поиск релевантной информации по запросу
- **Sort Source Service** - сортирует и ранжирует найденные источники
- **Context Service** - управляет историей диалога
- **LLM Service** - генерирует ответы с использованием языковых моделей

## Технологии

- FastAPI
- LangChain
- Redis
- Mistral AI/OpenRouter
- Tavily API

## Требования

- Python 3.8+
- Redis
- API ключи для:
  - Mistral AI/OpenRouter
  - Tavily

## Установка

1. Клонируйте репозиторий
2. Установите зависимости:
```bash
pip install -r requirements.txt
```
3. Создайте файл `.env` на основе `.env_example`:
```bash
MISTRAL_API_KEY=your_key
OPENROUTER_API_KEY=your_key
TAVILY_API_KEY=your_key
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

## Запуск

```bash
uvicorn app.main:app --reload
```

Не забудьте запкстить Redis локально.
После запуска API будет доступен по адресу: `http://localhost:8000`

## API Endpoints

### GET /
Приветственное сообщение и проверка работоспособности API.

### POST /chat
Основной эндпоинт для обработки запросов пользователей.

Пример запроса:
```json
{
    "chat_id": "123",
    "question": "Что такое мурабаха?"
}
```

Пример ответа:
```json
{
    "answer": "Подробный ответ о мурабахе...",
    "sources": ["url1", "url2"]
}
```

## Архитектура

1. Входящий запрос проверяется на соответствие тематике исламского финансирования
2. Если запрос валиден, выполняется поиск релевантной информации
3. Найденные источники сортируются по релевантности
4. Система получает историю диалога с пользователем
5. Генерируется ответ на основе контекста, источников и истории
6. Ответ и источники возвращаются пользователю

## Логирование

Система использует встроенный логгер для отслеживания всех операций и возможных ошибок.
