FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей (только необходимые)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей и установка
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY main.py virtual_world.py agent.py dqn.py env_const.py .

# Создание папки для сохранения моделей
RUN mkdir -p /app/models

# Запуск по умолчанию (обучение)
ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "train"]

