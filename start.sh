#!/bin/bash
set -e

# Start backend in background
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &

# Wait for backend to be ready
sleep 3

# Start frontend as main process (PID 1 via exec for proper signal handling)
exec streamlit run frontend/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless false \
    --server.enableXsrfProtection false \
    --server.enableCORS false \
    --server.maxUploadSize 25 \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false
