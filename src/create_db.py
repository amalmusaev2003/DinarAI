# run this script only for document addition to vector database
import os
import uuid
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from logger import logger
from config.settings import settings
from utils.llm_initializer import get_embedding_model

parser = argparse.ArgumentParser(description="Add file to vector database")
parser.add_argument("file_path", type=str, help='Input file path')
parser.add_argument(
    '--vectordb_path',
    type=str,
    default="chroma",
    help="Output vector database path"
)
args = parser.parse_args()

if not os.path.exists(args.file_path):
    print(f"Error: File '{args.file_path}' does not exist")
    exit(1)

file_to_vectorize = args.file_path
vectordb_path = args.vectordb_path

if not file_to_vectorize.endswith(".pdf"):
    logger.error("Only PDF files are supported")
    exit(1)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

logger.info(f"Adding {file_to_vectorize} file to vector database")

loader = PyPDFLoader(file_to_vectorize)
loaded_docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
    length_function=len
)
docs = text_splitter.split_documents(loaded_docs)
uuids = [str(uuid.uuid4()) for _ in range(len(docs))]

logger.info(f"Adding {len(docs)} chunks in vector database")

mistral_settings = settings.mistral
embeddings = get_embedding_model(mistral_settings.api_key)
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory=vectordb_path,
)

vector_store.add_documents(documents=docs, ids=uuids)

# example: python create_db.py "path\book.pdf" chroma