import pytest
import main
from main import app
from unittest.mock import MagicMock

@pytest.fixture
def client():
    app.config['TESTING'] = True
    
    # Mock audio and vision globally in main
    mock_audio = MagicMock()
    mock_vision = MagicMock()
    mock_vision.running = True
    mock_vision.navigation.get_status.return_value = {}
    mock_vision.diagnostics = {}
    mock_vision.latest_perception.scene_status = "Scanning..."
    mock_vision.latest_perception.ocr_text = ""
    mock_vision.latest_perception.navigation_context = ""
    
    main.get_audio = lambda: mock_audio
    main.get_vision = lambda: mock_vision

    with app.test_client() as client:
        yield client

def test_api_status(client):
    rv = client.get('/api/status')
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert "status" in json_data
    assert "scene_status" in json_data

def test_api_guardian(client):
    rv = client.post('/api/guardian', json={"name": "Alice", "phone": "1234567890"})
    assert rv.status_code == 200
    assert rv.get_json()["success"] is True

def test_api_sos_no_guardian(client):
    rv = client.post('/api/sos')
    assert rv.status_code == 200
    
def test_api_connect(client):
    # Already mocked as running=True
    rv = client.post('/api/connect')
    assert rv.status_code == 200
    assert rv.get_json()["success"] is True
