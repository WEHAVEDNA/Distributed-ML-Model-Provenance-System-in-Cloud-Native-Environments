FROM python:3.11-slim

WORKDIR /app

# Install system deps (for datasets C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# HuggingFace cache inside the container (mount a PVC here for persistence)
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
