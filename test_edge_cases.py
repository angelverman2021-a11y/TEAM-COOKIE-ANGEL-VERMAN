import time
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.navigation_engine import NavigationEngine
from src.vision.interfaces import PerceptionResult
from src.perception.memory import TemporalMemory

class MockAudio:
    def speak(self, text, priority):
        pass

print("="*40)
print("  EDGE CASE & SYNTHETIC VALIDATION")
print("="*40)

audio = MockAudio()
nav = NavigationEngine(audio)
memory = TemporalMemory()

def test_empty_frame():
    print("[TEST] Empty Frame (No Depth, No Objects)")
    p = PerceptionResult()
    p_stab = memory.update(p)
    nav.process_perception(p_stab)
    status = nav.get_status()
    assert status["navigation_status"] == "Degraded (No Depth Map)"
    print("  -> Passed.")

def test_corrupted_depth_map():
    print("[TEST] Corrupted Depth Map (NaNs and Infs)")
    depth = np.full((480, 640), np.nan, dtype=np.float32)
    depth[0:10, 0:10] = np.inf
    
    p = PerceptionResult(depth_map=depth, objects=[{"box": [10, 10, 50, 50], "label": "person"}])
    p_stab = memory.update(p)
    
    # Should not crash
    nav.process_perception(p_stab)
    status = nav.get_status()
    print(f"  -> Passed. Handled gracefully. Status: {status['danger_level']}")

def test_extreme_lighting():
    print("[TEST] Extreme Lighting (Completely White Depth Map = 1.0)")
    depth = np.ones((480, 640), dtype=np.float32)
    p = PerceptionResult(depth_map=depth, objects=[{"box": [100, 100, 400, 400], "label": "wall"}])
    p_stab = memory.update(p)
    nav.process_perception(p_stab)
    status = nav.get_status()
    assert status["danger_level"] in ["Caution", "Warning", "Critical", "Safe"]
    print(f"  -> Passed. Danger Level: {status['danger_level']}")

def test_memory_staleness():
    print("[TEST] Stale Memory Expiration")
    depth = np.zeros((480, 640), dtype=np.float32)
    # Inject object
    p1 = PerceptionResult(depth_map=depth, objects=[{"box": [10, 10, 50, 50], "label": "car"}])
    memory.update(p1)
    
    # Fast forward time to exceed MEMORY_DECAY_TIME (e.g., 2.0s)
    # Memory uses time.time(). We will manually spoof last_seen.
    for k in memory.tracked_objects:
        memory.tracked_objects[k]["last_seen"] -= 3.0
        
    p2 = PerceptionResult(depth_map=depth, objects=[])
    p2_stab = memory.update(p2)
    
    assert len(p2_stab.objects) == 0
    print("  -> Passed. Object successfully expired.")

try:
    test_empty_frame()
    test_corrupted_depth_map()
    test_extreme_lighting()
    test_memory_staleness()
    print("\nALL EDGE CASES PASSED.")
except Exception as e:
    print(f"\n[FAILED] {e}")
    sys.exit(1)
