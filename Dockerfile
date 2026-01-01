FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create non-root user
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

# Start the application
CMD ["python", "telegram_bot.py"]