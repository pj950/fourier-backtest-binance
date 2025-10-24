FROM python:3.11-slim

WORKDIR /app

RUN pip install poetry==2.2.1

COPY pyproject.toml ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY . .

RUN mkdir -p /app/data/cache

EXPOSE 8501

CMD ["streamlit", "run", "app/ui/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
