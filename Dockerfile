FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/yfinance_cache /app/data /app/charts

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=UTC
ENV YF_CACHE_LOCATION=/tmp/yfinance_cache

# Test critical imports
RUN python -c "import yfinance; print('✅ yfinance import successful')"
RUN python -c "from telegram import __version__; print(f'✅ python-telegram-bot {__version__}')"
RUN python -c "import pandas; print(f'✅ pandas {pandas.__version__}')"

# Run the bot
CMD ["python", "-u", "bot.py"]