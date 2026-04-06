FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only PyTorch to keep image size manageable (~700 MB).
# For GPU: replace the index-url with the CUDA wheel URL.
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV HF_HOME=/app/cache
# Increase uvicorn timeout for long-running /predict calls
ENV UVICORN_TIMEOUT_KEEP_ALIVE=120

EXPOSE 8003
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003", "--timeout-keep-alive", "120"]
