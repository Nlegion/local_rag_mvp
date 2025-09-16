import json
from typing import List, Dict
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from .data_processor import DataProcessor
from .vector_store import VectorStore
from .jira_client import MockJiraClient
from .logger import setup_logger

logger = setup_logger(__name__)


class RAGService:
    def __init__(self, data_dir: str = "data", persist_dir: str = "chroma_db"):
        self.data_processor = DataProcessor(data_dir)
        self.vector_store = VectorStore(
            embeddings=HuggingFaceEmbeddings(
                model_name="./models/LaBSE-en-ru",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            ),
            persist_dir=persist_dir
        )
        self.jira_client = MockJiraClient()

    def initialize(self):
        """Инициализирует RAG-систему."""
        try:
            logger.info("Инициализация RAG-системы")
            documents = self.data_processor.load_and_process_yaml()
            self.vector_store.build(documents)
            logger.info("RAG-система инициализирована успешно")
        except Exception as e:
            logger.error(f"Ошибка инициализации RAG-системы: {str(e)}")
            raise

    def load(self):
        """Загружает существующую RAG-систему."""
        try:
            logger.info("Загрузка RAG-системы")
            self.vector_store.load()
            logger.info("RAG-система загружена успешно")
        except Exception as e:
            logger.error(f"Ошибка загрузки RAG-системы: {str(e)}")
            raise

    def process_query(self, query: str) -> str:
        """Обрабатывает запрос пользователя и возвращает ответ."""
        try:
            logger.info(f"Обработка запроса: '{query}'")

            # Находим релевантные документы (категории)
            results = self.vector_store.search(query, k=3)
            if not results:
                return "Не могу найти информацию по вашему вопросу."

            # Берем наиболее релевантный документ (категорию)
            document = results[0]
            logger.info(f"Найдена категория: {document.metadata.get('category_id', 'Unknown')}")

            # Проверяем, есть ли системные действия в этом документе
            system_actions_str = document.metadata.get("system_actions", "[]")
            system_actions = json.loads(system_actions_str)

            # Выполняем системные действия (если есть)
            system_response = ""
            if system_actions:
                system_response = self.execute_system_actions(document, query)

            # Получаем инструкции для пользователя
            user_instructions = self.get_user_instructions(document)

            # Формируем итоговый ответ
            if system_response:
                # Если есть системный ответ, объединяем его с инструкциями
                response = f"{system_response}\n\n{user_instructions}"
            else:
                # Если системных действий нет, показываем только инструкции
                response = user_instructions

            return response

        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {str(e)}")
            return "Произошла ошибка при обработке вашего запроса."

    def execute_system_actions(self, document: Document, query: str) -> str:
        """Выполняет системные действия, указанные в документе."""
        try:
            system_actions_str = document.metadata.get("system_actions", "[]")
            system_actions = json.loads(system_actions_str)

            if not system_actions:
                return ""

            results = []
            for action in system_actions:
                action_type = action['action_config'].get('type', '')

                if action_type == 'create_jira_issue':
                    result = self.create_jira_issue(query, action['action_config'])
                    results.append(result)
                # Здесь можно добавить другие типы действий

            return "\n".join(results)

        except Exception as e:
            logger.error(f"Ошибка при выполнении системных действий: {str(e)}")
            return ""

    def get_user_instructions(self, document: Document) -> str:
        """Возвращает инструкции для пользователя из документа."""
        try:
            instructions_str = document.metadata.get("user_instructions", "[]")
            instructions = json.loads(instructions_str)

            if not instructions:
                return "Инструкции не найдены."

            # Форматируем инструкции в виде маркированного списка
            formatted_instructions = []
            for i, instruction in enumerate(instructions, 1):
                formatted_instructions.append(f"{i}. {instruction}")

            return "\n".join(formatted_instructions)

        except Exception as e:
            logger.error(f"Ошибка при получении инструкций: {str(e)}")
            return "Ошибка при получении инструкций."

    def create_jira_issue(self, user_query: str, action_config: Dict) -> str:
        """Создает задачу в Jira на основе конфигурации."""
        try:
            project_key = action_config.get("project_key", "PROC")
            issue_type = action_config.get("issue_type", "Task")
            summary_template = action_config.get("summary_template", "")
            description_template = action_config.get("description_template", "")

            summary = summary_template.format(user_query=user_query)
            description = description_template.format(user_query=user_query)

            components = action_config.get("components", [])
            labels = action_config.get("labels", [])

            # Создаем задачу в Jira
            issue_data = self.jira_client.create_issue(
                project_key=project_key,
                issue_type=issue_type,
                summary=summary,
                description=description,
                components=components,
                labels=labels
            )

            if issue_data:
                issue_key = issue_data.get("key", "UNKNOWN")
                issue_url = self.jira_client.get_issue_url(issue_key)
                return f"✅ Задача в Jira создана: {issue_key}. Ссылка: {issue_url}"
            else:
                return "❌ Не удалось создать задачу в Jira."

        except Exception as e:
            logger.error(f"Ошибка при создании задачи Jira: {str(e)}")
            return "❌ Ошибка при создании задачи в Jira."