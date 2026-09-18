FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-liberation fonts-dejavu-core libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000 8501

CMD ["python", "run.py", "api"]
