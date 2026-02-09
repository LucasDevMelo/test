FROM python:3.11-slim

WORKDIR /app/backend

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend/ /app/backend
COPY frontend/ /app/frontend

ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "profectus_ai.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
