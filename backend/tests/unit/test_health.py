from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_check_endpoint():
    """Test the health check endpoint returns 200 and expected status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "version": "1.0.0",
        "database": "connected"
    }

def test_root_endpoint():
    """Test the root endpoint returns 200 and expected message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Medical ICD Mapper API" in response.json()["message"]
