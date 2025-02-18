FROM python:3.12-slim

WORKDIR /app

# Poetry 설치
RUN pip install poetry

# Poetry 설정
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root

# 애플리케이션 코드 복사
COPY . .

EXPOSE 8000

# 실행
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
