FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Esponi la porta (obbligatorio per Render)
EXPOSE 8080

# Usa health_server.py invece di telegram_bot.py
CMD ["python", "health_server.py"]