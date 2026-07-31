import pytest
import time
import numpy as np
from src.navigation_engine import CollisionPredictor, ThreatDetector

class MockAudio:
    def __init__(self):
        self.logs = []
    def speak(self, text, priority):
        self.logs.append((priority, text))

def test_threat_detector():
    predictor = CollisionPredictor()
    audio = MockAudio()
    detector = ThreatDetector(audio)
    
    now = time.time()
    
    # Frame 1
    tracks = [{"track_id": 1, "label": "person", "dist": 0.4, "box": [0,0,10,10]}]
    col_data = predictor.update(tracks, now)
    detector.check_threats(tracks, col_data)
    
    # Frame 2
    now += 0.1
    tracks = [{"track_id": 1, "label": "person", "dist": 0.9, "box": [0,0,20,20]}] # Jumped by 0.5 in 0.1s -> v = 5.0
    col_data = predictor.update(tracks, now)
    detector.check_threats(tracks, col_data)
    
    assert col_data[1]["danger_level"] == "Critical"
    
    critical_logs = [l for l in audio.logs if l[0] == 1] # priority 1
    assert len(critical_logs) > 0
    assert "collision" in critical_logs[0][1].lower() or "close" in critical_logs[0][1].lower()
