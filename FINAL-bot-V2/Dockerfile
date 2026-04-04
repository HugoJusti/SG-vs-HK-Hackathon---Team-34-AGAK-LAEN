FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all bot files
COPY . .

# Create directories the bot writes to at runtime
RUN mkdir -p logs data bot_internal/logs

# Environment variables (overridden at runtime via --env-file or -e flags)
ENV RST_API_KEY=""
ENV RST_SECRET_KEY=""

# Run the bot
CMD ["python", "main-final.py"]
