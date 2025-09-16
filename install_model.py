import os
import sys
from huggingface_hub import snapshot_download, login

def download_model():
    model_name = "cointegrated/LaBSE-en-ru"
    local_dir = "./models/LaBSE-en-ru"
    print(f"Загрузка модели {model_name} в {local_dir}...")

    try:
        # Аутентификация (раскомментируйте, если есть токен)
        # login(token="YOUR_HF_TOKEN")

        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5"],  # Игнорируем ненужные файлы (если есть)
        )
        print("Модель успешно загружена!")
        print(f"Путь: {os.path.abspath(local_dir)}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
