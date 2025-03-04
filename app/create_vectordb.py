from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from config import EMBEDDING_MODEL, CHROMA_DB_PATH
from logging_config import logger

"""
Скрипт для загрузки документов в Chroma
"""
def load_documents_to_chroma():
    logger.info("Начало загрузки документов в Chroma")
    data_dir = os.getenv("DATA_PATH")
    if not os.path.exists(data_dir):
        logger.error(f"Папка {data_dir} не существует")
        raise ValueError(f"Папка {data_dir} не существует. Создайте её и добавьте документы.")

    documents = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith((".txt", ".md")):
            logger.debug(f"Загрузка файла: {filename}")
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())

    if not documents:
        logger.warning("Не найдено документов для загрузки")
        raise ValueError("Не найдено документов для загрузки.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    logger.info(f"Создано {len(docs)} чанков из документов")

    vector_store = Chroma.from_documents(
        docs,
        EMBEDDING_MODEL,
        persist_directory=CHROMA_DB_PATH
    )
    logger.info(f"Добавлено {len(docs)} чанков в векторную базу Chroma")

load_documents_to_chroma()