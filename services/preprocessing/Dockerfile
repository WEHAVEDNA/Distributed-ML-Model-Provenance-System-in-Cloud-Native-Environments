FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch CPU-only build keeps the image lean (~700 MB vs ~2 GB for CUDA)
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
