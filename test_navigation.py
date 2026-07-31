import pytest
import time
import numpy as np
from src.navigation_engine import NavigationEngine
from src.vision.interfaces import PerceptionResult

class MockAudioEngine:
    def __init__(self):
        self.logs = []
    
    def speak(self, msg, priority):
        self.logs.append((priority, msg))

def test_navigation_empty_perception():
    audio = MockAudioEngine()
    engine = NavigationEngine(audio)
    
    # Test empty perception
    perception = PerceptionResult()
    engine.process_perception(perception)
    
    status = engine.get_status()
    assert status["navigation_status"] == "Degraded (No Depth Map)"
    
def test_navigation_with_objects():
    audio = MockAudioEngine()
    engine = NavigationEngine(audio)
    
    depth_map = np.full((480, 640), 0.5, dtype=np.float32)
    
    objects = [
        {"box": [100, 100, 200, 200], "track_id": 1, "label": "person"}
    ]
    
    perception = PerceptionResult(
        objects=objects,
        depth_map=depth_map,
        timestamp=time.time()
    )
    
    engine.process_perception(perception)
    status = engine.get_status()
    
    assert status["navigation_status"] == "Active"
    assert status["nearest_obstacle"] == "person"
    assert status["obstacle_count"] == 0 # 0.5 is not > WARNING_DISTANCE (0.5)

def test_navigation_critical_threat():
    audio = MockAudioEngine()
    engine = NavigationEngine(audio)
    
    # Frame 1
    depth_map1 = np.full((480, 640), 0.2, dtype=np.float32)
    objects1 = [{"box": [100, 100, 200, 200], "track_id": 1, "label": "car"}]
    p1 = PerceptionResult(objects=objects1, depth_map=depth_map1, timestamp=time.time())
    engine.process_perception(p1)
    
    # Frame 2: Object rapidly approaches to depth 0.9 (Critical!)
    depth_map2 = np.full((480, 640), 0.9, dtype=np.float32)
    objects2 = [{"box": [50, 50, 250, 250], "track_id": 1, "label": "car"}]
    p2 = PerceptionResult(objects=objects2, depth_map=depth_map2, timestamp=time.time() + 0.1)
    engine.process_perception(p2)
    
    status = engine.get_status()
    assert status["danger_level"] == "Critical"
    assert status["collision_risk"] == "High"
    assert status["recommended_action"] == "Stop immediately"
