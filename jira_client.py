import os
import requests
from requests.auth import HTTPBasicAuth
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jira_client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()


class JiraClient:
    def __init__(self):
        try:
            self.jira_url = os.getenv("JIRA_URL", "https://your-company.atlassian.net")
            self.username = os.getenv("JIRA_USERNAME", "your-username")
            self.api_token = os.getenv("JIRA_API_TOKEN", "your-api-token")

            if not all([self.jira_url, self.username, self.api_token]):
                logger.warning("Не все переменные окружения для Jira заданы. Используется mock-режим.")
                raise ValueError("Missing Jira environment variables")

            self.auth = HTTPBasicAuth(self.username, self.api_token)
            self.headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            logger.info("JiraClient инициализирован с реальными учетными данными")

        except Exception as e:
            logger.warning(f"Ошибка инициализации JiraClient: {str(e)}. Используется MockJiraClient")
            self.use_mock = True
            self.mock_client = MockJiraClient()

    def create_issue(self, project_key: str, issue_type: str, summary: str,
                     description: str, components: list = None, labels: list = None) -> Optional[Dict[str, Any]]:
        """
        Создает задачу в Jira

        Args:
            project_key: Ключ проекта (например, "PROC")
            issue_type: Тип задачи (например, "Task", "Story")
            summary: Заголовок задачи
            description: Описание задачи
            components: Компоненты задачи
            labels: Метки задачи

        Returns:
            Dict с информацией о созданной задаче или None в случае ошибки
        """
        if hasattr(self, 'use_mock') and self.use_mock:
            return self.mock_client.create_issue(project_key, issue_type, summary, description, components, labels)

        # Формируем payload для Jira API
        payload = {
            "fields": {
                "project": {
                    "key": project_key
                },
                "issuetype": {
                    "name": issue_type
                },
                "summary": summary,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": description
                                }
                            ]
                        }
                    ]
                }
            }
        }

        # Добавляем компоненты, если указаны
        if components:
            payload["fields"]["components"] = [{"name": comp} for comp in components]

        # Добавляем метки, если указаны
        if labels:
            payload["fields"]["labels"] = labels

        try:
            logger.info(f"Попытка создания задачи в Jira: {summary}")
            response = requests.post(
                f"{self.jira_url}/rest/api/2/issue/",
                headers=self.headers,
                auth=self.auth,
                data=json.dumps(payload),
                timeout=30  # Таймаут 30 секунд
            )

            if response.status_code == 201:
                issue_data = response.json()
                logger.info(f"Задача создана успешно: {issue_data['key']}")
                return issue_data
            else:
                logger.error(f"Ошибка при создании задачи: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error("Таймаут при подключении к Jira API")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Ошибка подключения к Jira API")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при создании задачи: {str(e)}")
            return None

    def get_issue_url(self, issue_key: str) -> str:
        """Генерирует URL для просмотра задачи в Jira"""
        return f"{self.jira_url}/browse/{issue_key}"


class MockJiraClient:
    def __init__(self):
        self.issue_counter = 1
        self.issues = {}  # Для хранения "созданных" задач
        logger.info("MockJiraClient инициализирован")

    def create_issue(self, project_key: str, issue_type: str, summary: str,
                     description: str, components: list = None, labels: list = None) -> Optional[Dict[str, Any]]:
        """Имитирует создание задачи в Jira"""
        try:
            issue_key = f"{project_key}-{self.issue_counter}"
            self.issue_counter += 1

            # Сохраняем информацию о задаче
            self.issues[issue_key] = {
                "summary": summary,
                "description": description,
                "components": components,
                "labels": labels,
                "status": "Открыта",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Логируем "созданную" задачу
            logger.info(f"Mock Jira issue created: {issue_key}")
            logger.info(f"Summary: {summary}")
            logger.info(f"Description: {description[:100]}...")  # Логируем только начало описания

            return {
                "key": issue_key,
                "self": f"https://jira.example.com/rest/api/2/issue/{issue_key}",
                "id": str(self.issue_counter)
            }
        except Exception as e:
            logger.error(f"Ошибка в MockJiraClient: {str(e)}")
            return None

    def get_issue_url(self, issue_key: str) -> str:
        """Генерирует URL для просмотра задачи в Jira"""
        return f"https://jira.example.com/browse/{issue_key}"

    def get_issue_status(self, issue_key: str) -> Optional[str]:
        """Возвращает статус задачи (для тестирования)"""
        if issue_key in self.issues:
            return self.issues[issue_key]["status"]
        return None