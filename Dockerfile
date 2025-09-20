# Use Python 3.11 slim image as base for better performance
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Set environment variables for production deployment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    FLASK_DEBUG=false \
    PORT=5000 \
    NLTK_DATA=/usr/share/nltk_data \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    HF_HOME=/app/.cache/huggingface

# Install system dependencies required for ML libraries and enhanced features
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn[gevent]==21.2.0

# Download required NLTK data for text processing
RUN python -c "\
import nltk; \
nltk.download('punkt', download_dir='/usr/share/nltk_data'); \
nltk.download('stopwords', download_dir='/usr/share/nltk_data'); \
nltk.download('wordnet', download_dir='/usr/share/nltk_data')\
"

# Copy application code (excluding files in .dockerignore)
COPY . .

# Create necessary directories for models and cache
RUN mkdir -p models .cache/transformers .cache/huggingface logs

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' --shell /bin/bash appuser && \
    chown -R appuser:appuser /app && \
    chmod +x /app
USER appuser

# Expose the port that the app runs on
EXPOSE 5000

# Health check with better configuration for Cloud Run
HEALTHCHECK --interval=30s --timeout=15s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Use gunicorn for production deployment with Cloud Run optimized settings
CMD exec gunicorn \
     --bind 0.0.0.0:$PORT \
     --workers 2 \
     --worker-class gevent \
     --worker-connections 1000 \
     --timeout 300 \
     --keepalive 5 \
     --max-requests 1000 \
     --max-requests-jitter 100 \
     --preload \
     --access-logfile - \
     --error-logfile - \
     --log-level info \
     app:app