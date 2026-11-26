FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

COPY run_inference.py web_app.py ./
COPY src ./src
COPY models ./models

EXPOSE 8080

CMD ["uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
