from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document
from typing import List
from .logger import setup_logger

logger = setup_logger(__name__)


class VectorStore:
    def __init__(self, embeddings, persist_dir: str = "chroma_db"):
        self.embeddings = embeddings
        self.persist_dir = persist_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        self.vectorstore = None

    def build(self, documents: List[Document]):
        """Создает и наполняет векторную базу данных."""
        try:
            if not documents:
                error_msg = "Не предоставлены документы для построения векторной базы"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info("Разбиение документов на чанки...")
            chunks = self.text_splitter.split_documents(documents)

            # Фильтрация сложных метаданных
            logger.info("Фильтрация сложных метаданных...")
            filtered_chunks = filter_complex_metadata(chunks)
            logger.info(f"После фильтрации осталось чанков: {len(filtered_chunks)}")

            logger.info("Создание векторной базы данных...")
            self.vectorstore = Chroma.from_documents(
                documents=filtered_chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
            self.vectorstore.persist()
            logger.info(f"Векторная база создана и сохранена в '{self.persist_dir}'")

        except Exception as e:
            logger.error(f"Ошибка при построении векторной базы данных: {str(e)}")
            raise

    def load(self):
        """Загружает существующую векторную базу данных."""
        try:
            logger.info(f"Загрузка векторной базы из '{self.persist_dir}'")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            logger.info("Векторная база успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке векторной базы: {str(e)}")
            raise

    def search(self, query: str, k: int = 3):
        """Ищет k релевантных документов для запроса."""
        try:
            if self.vectorstore is None:
                self.load()

            logger.info(f"Поиск релевантных документов для: '{query}'")
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Найдено релевантных документов: {len(results)}")

            # Логируем найденные документы
            for i, doc in enumerate(results):
                logger.info(f"Документ {i + 1}: {doc.metadata.get('category_id', 'Unknown')} - {doc.metadata.get('category_description', 'No description')}")

            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске релевантных документов: {str(e)}")
            return []