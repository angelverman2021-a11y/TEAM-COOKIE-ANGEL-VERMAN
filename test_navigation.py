import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.navigation_engine import NavigationEngine, NavigationManager
from src.audio_engine import AudioEngine

print("="*60)
print("       TESTING NAVIGATION ENGINE")
print("="*60)

try:
    print("[TEST 1] Initializing NavigationEngine (MiDaS)...")
    engine = NavigationEngine()
    if not engine.active:
        raise RuntimeError("NavigationEngine failed to initialize.")
        
    print("[TEST 2] Running Depth Inference...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_map = engine.get_depth(dummy_frame)
    print(f"-> Depth map generated: shape={depth_map.shape}, max={depth_map.max():.2f}")
    
    print("\n[TEST 3] NavigationManager integration...")
    # Mocking VisionEngine
    class MockVision:
        def __init__(self):
            class MockCameraManager:
                def get_frame(self):
                    return dummy_frame
            self.camera_manager = MockCameraManager()
            import threading
            self.ai_lock = threading.Lock()
            self.latest_tracked = [{"box": [100, 100, 300, 300], "conf": 0.9, "cls": 0, "track_id": 1, "label": "person"}]

    audio = AudioEngine()
    vision = MockVision()
    nav_mgr = NavigationManager(audio, vision)
    nav_mgr.start()
    
    import time
    time.sleep(2.0) # Let it run a few loops
    
    status = nav_mgr.get_status()
    print(f"-> Status: {status}")
    nav_mgr.stop()
    
    print("\nALL NAVIGATION TESTS PASSED!")
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)
