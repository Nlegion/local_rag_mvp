import os
import yaml
import json
from typing import List, Dict
from langchain.schema import Document
from .logger import setup_logger

logger = setup_logger(__name__)


class DataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def load_and_process_yaml(self) -> List[Document]:
        """Загружает и парсит все YAML-файлы в директории, создавая отдельные документы для каждой категории."""
        documents = []
        try:
            if not os.path.exists(self.data_dir):
                logger.error(f"Директория с данными не существует: {self.data_dir}")
                return documents

            files = os.listdir(self.data_dir)
            yaml_files = [f for f in files if f.endswith((".yaml", ".yml"))]

            if not yaml_files:
                logger.warning(f"В директории {self.data_dir} не найдено YAML-файлов")
                return documents

            logger.info(f"Найдено YAML-файлов: {len(yaml_files)}: {', '.join(yaml_files)}")

            for filename in yaml_files:
                try:
                    file_path = os.path.join(self.data_dir, filename)
                    logger.info(f"Обработка файла: {filename}")

                    with open(file_path, 'r', encoding='utf-8') as file:
                        yaml_data = yaml.safe_load(file)

                    # Обрабатываем каждую категорию как отдельный документ
                    categories = yaml_data.get('categories', [])
                    for category in categories:
                        # Формируем текст для поиска, специфичный для этой категории
                        process_text = f"""
                        Процесс: {yaml_data.get('metadata', {}).get('process_name', 'Название не указано')}
                        Категория: {category.get('category_id', 'N/A')} - {category.get('description', 'Описание отсутствует')}
                        Триггеры: {", ".join(yaml_data.get('triggers', []))}
                        """

                        system_actions = []
                        user_instructions = []

                        # Обрабатываем шаги категории
                        steps = category.get('steps', [])
                        for step in steps:
                            step_desc = step.get('description', 'Описание отсутствует')
                            process_text += f"Шаг: {step_desc}\n"

                            # Собираем системные действия
                            if 'system_action' in step:
                                system_actions.append({
                                    'step_description': step_desc,
                                    'action_config': step['system_action']
                                })

                            # Собираем инструкции для пользователя
                            if 'user_instruction' in step:
                                user_instructions.append(step['user_instruction'])

                        # Создаем документ с метаданными для этой категории
                        metadata = {
                            "source": filename,
                            "category_id": category.get('category_id', ''),
                            "category_description": category.get('description', ''),
                            "system_actions": json.dumps(system_actions),
                            "user_instructions": json.dumps(user_instructions),
                            "has_system_actions": len(system_actions) > 0
                        }

                        doc = Document(page_content=process_text, metadata=metadata)
                        documents.append(doc)
                        logger.debug(f"Категория {category.get('category_id')} обработана, инструкций: {len(user_instructions)}")

                except Exception as e:
                    logger.error(f"Ошибка обработки файла {filename}: {str(e)}")

            logger.info(f"Всего создано документов (категорий): {len(documents)}")
            return documents

        except Exception as e:
            logger.error(f"Ошибка в load_and_process_yaml: {str(e)}")
            return []