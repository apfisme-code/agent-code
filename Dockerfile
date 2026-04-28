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
COPY . .

# Создание папки для сохранения моделей
RUN mkdir -p /app/models

# Запуск по умолчанию (обучение и проверка)
ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "train_and_test"]

