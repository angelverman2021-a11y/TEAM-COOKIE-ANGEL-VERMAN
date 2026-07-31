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
    
    main.get_audio = lambda: mock_audio
    main.get_vision = lambda: mock_vision

    with app.test_client() as client:
        yield client

def test_frontend_index(client):
    rv = client.get('/')
    assert rv.status_code == 200
    html = rv.data.decode('utf-8')
    assert "id=\"btn-connect\"" in html or "id=\"btn-connect-card\"" in html
    # Assume the page loads ok.
