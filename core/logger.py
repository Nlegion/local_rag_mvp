# core/logger.py
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Создаем папку для логов, если она не существует
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Создаем форматтер для логов
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Настраивает и возвращает логгер с указанным именем и уровнем логирования.

    Args:
        name: Имя логгера (обычно __name__)
        log_level: Уровень логирования (по умолчанию INFO)

    Returns:
        Настроенный логгер
    """
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Если у логгера уже есть обработчики, не добавляем новые
    if logger.handlers:
        return logger

    # Создаем обработчик для записи в файл с ротацией
    log_filename = f"{LOG_DIR}/rag_system_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)

    # Создаем обработчик для вывода в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Создаем форматтер и добавляем его к обработчикам
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Добавляем обработчики к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Создаем корневой логгер для всего приложения
app_logger = setup_logger("rag_system")