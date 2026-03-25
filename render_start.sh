#!/bin/bash
# 1. Start the FastAPI Backend in the background
# We run it on port 8000 internally.
echo "🚀 Starting Medical ICD Mapper API (Backend)..."
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 > backend_logs.txt 2>&1 &

# 2. Give the backend a few seconds to initialize
echo "⏳ Waiting for backend to initialize..."
sleep 10

# 3. Start the Streamlit Dashboard (Frontend)
# This will bind to the $PORT Render provides to the outside world.
echo "🏥 Starting Medical ICD Mapper Dashboard (Frontend)..."
streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
