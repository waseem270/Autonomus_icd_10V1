import streamlit as st
import subprocess
import time
import requests
import pytest

def test_frontend_imports():
    """Test that frontend imports work."""
    try:
        import frontend.app
        import frontend.utils
    except Exception as e:
        pytest.fail(f"Frontend imports failed: {e}")

def test_backend_health_reachable():
    """Test that the backend is reachable (assuming it's running via uvicorn)."""
    # Note: This requires the backend to be running in another process.
    # The terminal shows it should be running on localhost:8000.
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    except Exception:
        # If it's not running, we skip this test or just fail it.
        # But wait, uvicorn is running in another terminal.
        pytest.fail("Backend not reachable at http://localhost:8000/health")
