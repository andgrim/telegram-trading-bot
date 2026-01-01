FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies as root (sopprime warning)
ENV PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user for runtime
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

# Start the application
CMD ["python", "telegram_bot.py"]