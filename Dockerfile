FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860

# Install system dependencies required for scientific computing and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and checkpoint
COPY . .

# Expose FastAPI application port
EXPOSE 7860

# Add container health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Launch FastAPI ASGI server
CMD ["uvicorn", "src.inference.predict:app", "--host", "0.0.0.0", "--port", "7860"]
