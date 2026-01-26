FROM python:3.11-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create cache directory
RUN mkdir -p /tmp/yfinance_cache

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Test critical imports
RUN python -c "import yfinance; print('✅ yfinance import successful')"

# Run the bot
CMD ["python", "-u", "bot.py"]