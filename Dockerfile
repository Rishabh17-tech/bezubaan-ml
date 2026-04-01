# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copy code + model + labels
COPY main.py .
COPY requirements.txt .
COPY animal_classifier_model.h5 .
COPY labels.json .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create temp folder
RUN mkdir temp

# Expose port for Render
ENV PORT=10000

# Run app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]