# Use Python 3.11 as base image
FROM python:3.11-slim

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install system dependencies (OCR, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR $HOME/app

# Copy requirements and install as user
COPY --chown=user requirements.txt .
USER user
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Create required folders
RUN mkdir -p uploads logs database

# Expose port (HF Spaces expects 7860)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Command to run both services
CMD python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 & \
    sleep 5 && \
    streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
