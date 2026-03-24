FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    "numpy<2" \
    fastapi==0.110.* \
    uvicorn==0.29.* \
    boto3==1.34.* \
    transformers==4.40.* \
    accelerate==0.29.*
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
COPY main.py .
EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
