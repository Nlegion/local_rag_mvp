# rag_core.py
import os
import yaml
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Конфигурация путей
MODEL_PATH = "./models/model-q4_K.gguf"
DATA_DIR = "data"
PERSIST_DIR = "chroma_db"


class RAGSystem:
    def __init__(self):
        # Инициализация модели для эмбеддингов (легкая, работает на CPU)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

        # Инициализация текстового сплиттера
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # Callback менеджер для потокового вывода
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # Инициализация локальной LLM (Saiga 3 8B)
        self.llm = LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.1,
            max_tokens=2000,
            top_p=1,
            callback_manager=callback_manager,
            verbose=False,
            n_gpu_layers=0,  # Для CPU используйте 0. Для GPU укажите количество слоев.
            n_ctx=4096,  # Размер контекста
            n_batch=512,  # Размер батча для обработки
            f16_kv=True,  # Использование половинной точности для кэша ключей/значений
        )

        self.vectorstore = None

    def load_and_process_yaml(self) -> List[Document]:
        """Загружает и парсит все YAML-файлы в директории, преобразуя их в текстовые документы."""
        documents = []
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                file_path = os.path.join(DATA_DIR, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    yaml_data = yaml.safe_load(file)

                # Преобразование структуры YAML в читаемый текст для поиска
                process_text = f"""
                Процесс: {yaml_data.get('metadata', {}).get('process_name', 'Название не указано')}
                Владелец: {yaml_data.get('metadata', {}).get('process_owner', 'Не указан')}
                Версия: {yaml_data.get('metadata', {}).get('version', 'Не указана')}

                Триггеры: {", ".join(yaml_data.get('triggers', []))}

                Категории:
                """
                for category in yaml_data.get('categories', []):
                    process_text += f"\nКатегория {category['category_id']}: {category['description']}\n"
                    for step in category.get('steps', []):
                        process_text += f"  Шаг {step['step_id']}: {step['description']}\n"
                        if 'link' in step:
                            process_text += f"    Ссылка: {step['link']}\n"

                # Создание документа LangChain с метаданными об источнике
                metadata = {"source": filename}
                documents.append(Document(page_content=process_text, metadata=metadata))
        return documents

    def build_vector_store(self):
        """Создает и наполняет векторную базу данных."""
        print("Загрузка и обработка YAML-документов...")
        docs = self.load_and_process_yaml()
        if not docs:
            raise ValueError(f"Не найдено YAML-файлов в директории {DATA_DIR}")

        print("Разбиение документов на чанки...")
        chunks = self.text_splitter.split_documents(docs)

        print("Создание векторной базы данных...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=PERSIST_DIR
        )
        self.vectorstore.persist()
        print(f"Векторная база создана и сохранена в '{PERSIST_DIR}'")

    def load_vector_store(self):
        """Загружает существующую векторную базу данных."""
        self.vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=self.embeddings
        )

    def get_relevant_context(self, query: str, k: int = 3) -> List[Document]:
        """Ищет k релевантных чанков для запроса."""
        if self.vectorstore is None:
            self.load_vector_store()
        return self.vectorstore.similarity_search(query, k=k)

    def generate_answer(self, query: str) -> str:
        """Генерирует ответ на вопрос, используя контекст из векторной БД."""
        relevant_docs = self.get_relevant_context(query)
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in relevant_docs]))

        # Создание промта с четким указанием контекста и инструкций
        prompt = f"""
        <контекст>
        {context_text}
        </контекст>

        Используй только приведенную выше информацию, чтобы точно ответить на следующий вопрос.
        Если ответ не содержится в контексте, вежливо скажи, что не знаешь ответа.
        Отвечай кратко, ясно и по делу. Ответ должен быть на русском языке.

        Вопрос: {query}

        Ответ:
        """
        response = self.llm.invoke(prompt)
        response += f"\n\nИсточники: {', '.join(sources)}"
        return response


# Утилита для переиндексации данных
def reindex_data():
    rag = RAGSystem()
    rag.build_vector_store()


if __name__ == "__main__":
    reindex_data()