FROM python:3.11-slim

# ffmpeg + fonts (for burned-in captions via libass) + certs for HTTPS downloads
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /app/data/storage /app/data/tmp

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
