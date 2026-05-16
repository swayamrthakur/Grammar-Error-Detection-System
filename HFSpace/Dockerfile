FROM python:3.10-slim

# Install Java (required for LanguageTool)
RUN apt-get update && apt-get install -y \
    default-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860

CMD gunicorn --bind 0.0.0.0:7860 --timeout 120 app:app