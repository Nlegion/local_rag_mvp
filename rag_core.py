import os
import yaml
import json
from typing import List, Dict
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from jira_client import MockJiraClient
import logging

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

# Конфигурация путей
MODEL_PATH = "./models/model-q4_K.gguf"
DATA_DIR = "data"
PERSIST_DIR = "chroma_db"


class RAGSystem:
    def __init__(self):
        logger.info("Инициализация RAGSystem")
        try:
            # Инициализация модели для эмбеддингов
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

            # Инициализация клиента Jira
            self.jira_client = MockJiraClient()
            self.vectorstore = None
            logger.info("RAGSystem инициализирован успешно")

        except Exception as e:
            logger.error(f"Ошибка инициализации RAGSystem: {str(e)}")
            raise

    def load_and_process_yaml(self) -> List[Document]:
        """Загружает и парсит все YAML-файлы в директории."""
        documents = []
        try:
            if not os.path.exists(DATA_DIR):
                logger.error(f"Директория с данными не существует: {DATA_DIR}")
                return documents

            files = os.listdir(DATA_DIR)
            yaml_files = [f for f in files if f.endswith((".yaml", ".yml"))]

            if not yaml_files:
                logger.warning(f"В директории {DATA_DIR} не найдено YAML-файлов")
                return documents

            logger.info(f"Найдено YAML-файлов: {len(yaml_files)}: {', '.join(yaml_files)}")

            for filename in yaml_files:
                try:
                    file_path = os.path.join(DATA_DIR, filename)
                    logger.info(f"Обработка файла: {filename}")

                    with open(file_path, 'r', encoding='utf-8') as file:
                        yaml_data = yaml.safe_load(file)

                    # Формируем текст для поиска
                    process_text = f"""
                    Процесс: {yaml_data.get('metadata', {}).get('process_name', 'Название не указано')}
                    Триггеры: {", ".join(yaml_data.get('triggers', []))}
                    """

                    system_actions = []
                    user_instructions = []

                    # Обрабатываем категории и шаги
                    categories = yaml_data.get('categories', [])
                    for category in categories:
                        process_text += f"\nКатегория {category.get('category_id', 'N/A')}: {category.get('description', 'Описание отсутствует')}\n"
                        steps = category.get('steps', [])
                        for step in steps:
                            step_desc = step.get('description', 'Описание отсутствует')
                            process_text += f"  - {step_desc}\n"

                            # Собираем системные действия
                            if 'system_action' in step:
                                system_actions.append({
                                    'step_description': step_desc,
                                    'action_config': step['system_action']
                                })

                            # Собираем инструкции для пользователя
                            if 'user_instruction' in step:
                                user_instructions.append(step['user_instruction'])

                    # Создаем документ с метаданными
                    metadata = {
                        "source": filename,
                        "system_actions": json.dumps(system_actions),
                        "user_instructions": json.dumps(user_instructions),
                        "has_system_actions": len(system_actions) > 0
                    }

                    doc = Document(page_content=process_text, metadata=metadata)
                    documents.append(doc)
                    logger.debug(f"Файл {filename} обработан, системных действий: {len(system_actions)}")

                except Exception as e:
                    logger.error(f"Ошибка обработки файла {filename}: {str(e)}")

            return documents

        except Exception as e:
            logger.error(f"Ошибка в load_and_process_yaml: {str(e)}")
            return []

    def build_vector_store(self):
        """Создает и наполняет векторную базу данных."""
        try:
            logger.info("Начало построения векторной базы данных")
            docs = self.load_and_process_yaml()
            if not docs:
                error_msg = f"Не найдено YAML-файлов в директории {DATA_DIR} или они содержат ошибки"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info("Разбиение документов на чанки...")
            chunks = self.text_splitter.split_documents(docs)

            # Фильтрация сложных метаданных
            logger.info("Фильтрация сложных метаданных...")
            filtered_chunks = filter_complex_metadata(chunks)
            logger.info(f"После фильтрации осталось чанков: {len(filtered_chunks)}")

            logger.info("Создание векторной базы данных...")
            self.vectorstore = Chroma.from_documents(
                documents=filtered_chunks,
                embedding=self.embeddings,
                persist_directory=PERSIST_DIR
            )
            self.vectorstore.persist()
            logger.info(f"Векторная база создана и сохранена в '{PERSIST_DIR}'")

        except Exception as e:
            logger.error(f"Ошибка при построении векторной базы данных: {str(e)}")
            raise

    def load_vector_store(self):
        """Загружает существующую векторную базу данных."""
        try:
            logger.info(f"Загрузка векторной базы из '{PERSIST_DIR}'")
            self.vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=self.embeddings
            )
            logger.info("Векторная база успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке векторной базы: {str(e)}")
            raise

    def get_relevant_document(self, query: str) -> Document:
        """Находит наиболее релевантный документ для запроса."""
        try:
            if self.vectorstore is None:
                self.load_vector_store()

            logger.info(f"Поиск релевантного документа для: '{query}'")
            results = self.vectorstore.similarity_search(query, k=3)

            if not results:
                logger.warning("Не найдено релевантных документов")
                return None

            # Возвращаем наиболее релевантный документ
            return results[0]

        except Exception as e:
            logger.error(f"Ошибка при поиске документа: {str(e)}")
            return None

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

    def process_query(self, query: str) -> str:
        """Обрабатывает запрос пользователя и возвращает ответ."""
        try:
            logger.info(f"Обработка запроса: '{query}'")

            # Находим релевантный документ
            document = self.get_relevant_document(query)
            if not document:
                return "Не могу найти информацию по вашему вопросу."

            # Выполняем системные действия (если есть)
            system_response = self.execute_system_actions(document, query)

            # Получаем инструкции для пользователя
            user_instructions = self.get_user_instructions(document)

            # Формируем итоговый ответ
            response = user_instructions
            if system_response:
                response += f"\n\n{system_response}"

            return response

        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {str(e)}")
            return "Произошла ошибка при обработке вашего запроса."


# Утилита для переиндексации данных
def reindex_data():
    try:
        logger.info("Запуск переиндексации данных")
        rag = RAGSystem()
        rag.build_vector_store()
        logger.info("Переиндексация данных завершена успешно")
    except Exception as e:
        logger.error(f"Ошибка при переиндексации данных: {str(e)}")
        raise


def force_reindex():
    """Принудительная переиндексация данных с удалением старой базы"""
    import shutil
    try:
        # Удаляем старую базу данных
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
            logger.info(f"Удалена старая векторная база: {PERSIST_DIR}")

        # Создаем новую базу
        reindex_data()
        logger.info("Принудительная переиндексация завершена успешно")
        return True
    except Exception as e:
        logger.error(f"Ошибка при принудительной переиндексации: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='RAG System Management')
    parser.add_argument('--reindex', action='store_true', help='Принудительная переиндексация данных')
    args = parser.parse_args()

    if args.reindex:
        force_reindex()
    else:
        reindex_data()