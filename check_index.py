import os
from rag_core import RAGSystem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_indexed_documents():
    """Проверяет, какие документы загружены в векторную базу"""
    rag = RAGSystem()
    rag.load_vector_store()

    # Получаем все документы из базы
    collection = rag.vectorstore._collection
    documents = collection.get(include=["metadatas", "documents"])

    logger.info(f"Всего документов в базе: {len(documents['ids'])}")

    for i, (metadata, doc_content) in enumerate(zip(documents['metadatas'], documents['documents'])):
        logger.info(f"Документ {i + 1}:")
        logger.info(f"  Источник: {metadata.get('source', 'Unknown')}")
        logger.info(f"  Длина контента: {len(doc_content)} символов")
        logger.info(f"  Первые 200 символов: {doc_content[:200]}...")
        logger.info("---")


if __name__ == "__main__":
    check_indexed_documents()