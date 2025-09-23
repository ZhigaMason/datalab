FROM apache/spark-py:latest

USER root
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

USER 185
ENV PYTHONUNBUFFERED=1

CMD ["python3", "/app/main.py"]
