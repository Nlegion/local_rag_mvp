import re
from jira_client import JiraClient, MockJiraClient
import os
import logging
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

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
            max_tokens=1000,
            top_p=1,
            callback_manager=callback_manager,
            verbose=False,
            n_gpu_layers=0,
            n_ctx=4096,
            n_batch=512,
            f16_kv=True,
            stop=["\n\n", "Источники:", "ИСТОЧНИКИ:"],
        )

        self.vectorstore = None
        #self.jira_client = JiraClient()  # Добавляем клиент Jira
        self.jira_client = MockJiraClient()

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

    # rag_core.py (улучшенная обработка ошибок)
    def generate_answer(self, query: str) -> str:
        """Генерирует ответ на вопрос, используя контекст из векторной БД"""
        try:
            relevant_docs = self.get_relevant_context(query)
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in relevant_docs]))

            # Проверяем, относится ли запрос к групповой закупке
            jira_response = ""
            if self.is_group_procurement_query(query, context_text):
                jira_response = self.create_procurement_issue(query)

            # Улучшенный промт
            prompt = f"""
            Ты - помощник по внутренним процессам компании. Ответь на вопрос пользователя, используя только предоставленную информацию.

            Контекстная информация:
            {context_text}

            Инструкции:
            1. Отвечай только на основе предоставленной информации
            2. Если информации для ответа нет, скажи "Не могу найти информацию по этому вопросу"
            3. Будь кратким и конкретным
            4. Отвечай на русском языке
            5. Не придумывай информацию, которой нет в контексте

            Вопрос: {query}

            Ответ:
            """

            response = self.llm.invoke(prompt)

            # Очищаем ответ от возможного мусора
            response = self.clean_response(response)

            # Если ответ пустой или слишком короткий, используем fallback
            if len(response.strip()) < 10:
                response = "Не могу найти точную информацию по вашему вопросу. Пожалуйста, обратитесь к соответствующему отделу."

            # Добавляем информацию о задаче Jira, если она была создана
            if jira_response:
                response += f"\n\n{jira_response}"

            response += f"\n\nИсточники: {', '.join(sources)}"
            return response

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {str(e)}")
            return f"Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.\n\nОшибка: {str(e)}"

    def clean_response(self, response: str) -> str:
        """Очищает ответ от бессмысленного текста и повторений"""
        # Удаляем все после двойного перевода строки
        if "\n\n" in response:
            response = response.split("\n\n")[0]

        # Удаляем числа и пункты списка в конце, которые могут быть артефактами генерации
        lines = response.split('\n')
        clean_lines = []

        for line in lines:
            # Пропускаем пустые строки и строки с одним символом
            if len(line.strip()) <= 1:
                continue

            # Пропускаем строки, которые выглядят как артефакты генерации
            if re.match(r'^(\d+\.?|•|\-|\*)\s*$', line.strip()):
                continue

            # Пропускаем строки, содержащие только специальные символы
            if re.match(r'^[#*_\-→⇒⇨›»>]+$', line.strip()):
                continue

            clean_lines.append(line)

        response = '\n'.join(clean_lines)

        # Обрезаем ответ до последнего осмысленного предложения
        sentences = re.split(r'(?<=[.!?])\s+', response)
        if len(sentences) > 1:
            # Ищем последнее законченное предложение
            last_complete_sentence = None
            for i in range(len(sentences) - 1, -1, -1):
                if re.search(r'[.!?]$', sentences[i]):
                    last_complete_sentence = i
                    break

            if last_complete_sentence is not None:
                response = ' '.join(sentences[:last_complete_sentence + 1])

        return response.strip()

    def is_group_procurement_query(self, query: str, context: str) -> bool:
        """Определяет, относится ли запрос к групповой закупке"""
        procurement_keywords = [
            "групповая закупка", "коллективная закупка",
            "group purchasing", "gpo", "организовать закупку",
            "массовая закупка", "совместная закупка", "закупка для отдела",
            "закупка для команды", "централизованная закупка"
        ]

        # Проверяем наличие ключевых слов в запросе
        query_lower = query.lower()

        for keyword in procurement_keywords:
            if keyword in query_lower:
                return True

        # Дополнительная проверка по контексту
        context_lower = context.lower()
        if any(keyword in context_lower for keyword in procurement_keywords):
            return True

        return False

    def create_procurement_issue(self, user_query: str) -> str:
        """Создает задачу в Jira для групповой закупки:cite[1]"""
        # Здесь можно добавить логику для извлечения деталей из запроса
        # Пока используем шаблонные значения
        project_key = "PROC"
        issue_type = "Task"
        summary = f"Групповая закупка: запрос от пользователя"
        description = f"""Пользователь запросил организацию групповой закупки.

Детали запроса:
{user_query}

Необходимо:
1. Связаться с пользователем для уточнения деталей
2. Провести анализ рынка
3. Найти потенциальных участников закупки
4. Организовать переговоры с поставщиками
5. Согласовать условия поставки"""

        # Создаем задачу в Jira
        issue_data = self.jira_client.create_issue(
            project_key=project_key,
            issue_type=issue_type,
            summary=summary,
            description=description,
            components=["Закупки"],
            labels=["group_purchasing", "procurement"]
        )

        if issue_data:
            issue_key = issue_data.get("key", "UNKNOWN")
            issue_url = self.jira_client.get_issue_url(issue_key)
            return f"✅ Задача в Jira создана успешно: {issue_key}. Ссылка: {issue_url}"
        else:
            return "❌ Не удалось создать задачу в Jira. Пожалуйста, свяжитесь с отделом закупок напрямую."


# Утилита для переиндексации данных
def reindex_data():
    rag = RAGSystem()
    rag.build_vector_store()


if __name__ == "__main__":
    reindex_data()
