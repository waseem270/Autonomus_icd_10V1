# Use Python 3.11 as base image
FROM python:3.11-slim

# Install system dependencies (OCR, etc.)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create dummy folders for persistence (Hugging Face Spaces are ephemeral but helpful for logic)
RUN mkdir -p uploads logs database

# Expose ports (FastAPI on 8000, Streamlit on 7860 for HF)
EXPOSE 8000
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Command to run both services
# Note: Hugging Face listens on 7860 by default for Streamlit/Gradio
CMD python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
