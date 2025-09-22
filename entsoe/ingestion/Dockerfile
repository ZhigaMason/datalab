FROM apache/spark-py:latest
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python3", "/app/main.py"]
