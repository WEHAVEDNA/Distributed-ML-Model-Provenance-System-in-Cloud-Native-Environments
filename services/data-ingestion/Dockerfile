FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    fastapi==0.110.* \
    uvicorn==0.29.* \
    boto3==1.34.* \
    datasets==2.19.* \
    huggingface-hub==0.22.*
COPY main.py .
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
