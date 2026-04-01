FROM python:3.10-slim

WORKDIR /app

COPY main.py .
COPY requirements.txt .
COPY animal_classifier_model.keras .
COPY labels.json .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir temp

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]