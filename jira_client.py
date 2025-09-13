import os
import requests
from requests.auth import HTTPBasicAuth
import json
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()


class JiraClient:
    def __init__(self):
        self.jira_url = os.getenv("JIRA_URL", "https://your-company.atlassian.net")
        self.username = os.getenv("JIRA_USERNAME", "your-username")
        self.api_token = os.getenv("JIRA_API_TOKEN", "your-api-token")
        self.auth = HTTPBasicAuth(self.username, self.api_token)
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def create_issue(self, project_key: str, issue_type: str, summary: str,
                     description: str, components: list = None, labels: list = None) -> Optional[Dict[str, Any]]:
        """
        Создает задачу в Jira:cite[1]:cite[7]

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
        # Формируем payload для Jira API:cite[1]
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
            response = requests.post(
                f"{self.jira_url}/rest/api/2/issue/",
                headers=self.headers,
                auth=self.auth,
                data=json.dumps(payload)
            )

            if response.status_code == 201:
                issue_data = response.json()
                logger.info(f"Задача создана успешно: {issue_data['key']}")
                return issue_data
            else:
                logger.error(f"Ошибка при создании задачи: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Исключение при создании задачи: {str(e)}")
            return None

    def get_issue_url(self, issue_key: str) -> str:
        """Генерирует URL для просмотра задачи в Jira"""
        return f"{self.jira_url}/browse/{issue_key}"


class MockJiraClient:
    def __init__(self):
        self.issue_counter = 1
        self.issues = {}  # Для хранения "созданных" задач

    def create_issue(self, project_key: str, issue_type: str, summary: str,
                     description: str, components: list = None, labels: list = None) -> Optional[Dict[str, Any]]:
        """Имитирует создание задачи в Jira"""
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
        logger.info(f"Description: {description}")

        return {
            "key": issue_key,
            "self": f"https://jira.example.com/rest/api/2/issue/{issue_key}",
            "id": str(self.issue_counter)
        }

    def get_issue_url(self, issue_key: str) -> str:
        """Генерирует URL для просмотра задачи в Jira"""
        return f"https://jira.example.com/browse/{issue_key}"

    def get_issue_status(self, issue_key: str) -> Optional[str]:
        """Возвращает статус задачи (для тестирования)"""
        if issue_key in self.issues:
            return self.issues[issue_key]["status"]
        return None