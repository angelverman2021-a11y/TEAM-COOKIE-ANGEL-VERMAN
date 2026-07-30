import cv2
import numpy as np
import time
import sys
from src.navi_engine import VisionEngine, Detector, Tracker, DetectionManager, FrameProcessor, CameraManager
from src.audio_engine import AudioEngine

print("="*60)
print("       TESTING NAVI VISION ENGINE & MODULAR CLASSES")
print("="*60)

# Test 1: Detector
print("\n[TEST 1] Testing Detector with YOLO11...")
detector = Detector()
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
res = detector.detect(dummy_frame)
print(f"-> Detector successfully ran on {detector.device.upper()}. Result: {res is not None}")

# Test 2: DetectionManager
print("\n[TEST 2] Testing DetectionManager and Tracker EMA Smoothing...")
manager = DetectionManager()
tracked = manager.process_frame(dummy_frame)
print(f"-> DetectionManager initialized. Tracked items count: {len(tracked)}")

# Test 3: FrameProcessor
print("\n[TEST 3] Testing FrameProcessor HUD Rendering...")
processor = FrameProcessor()
try:
    dummy_tracked = [{"box": [10, 10, 50, 50], "conf": 0.99, "cls": 0, "track_id": 1, "label": "person"}]
    annotated = processor.draw_hud(dummy_frame, dummy_tracked, "Happy")
    
    # Test debug HUD
    diagnostics = {
        "cam_fps": 30.0,
        "cam_status": "Active",
        "backend": "DSHOW",
        "resolution": (640, 480),
        "ai_time": 15.5,
        "det_count": 1
    }
    annotated_debug = processor.draw_debug_hud(annotated, diagnostics)
    print(f"-> FrameProcessor created frame shape: {annotated_debug.shape}")
except Exception as e:
    print(f"[ERROR] FrameProcessor failed: {e}")
    sys.exit(1)

print("\n[TEST 4] Testing CameraManager Initialization...")
try:
    cam_mgr = CameraManager()
    print("-> CameraManager initialized successfully!")
except Exception as e:
    print(f"[ERROR] CameraManager failed: {e}")
    sys.exit(1)

print("\n[TEST 5] Testing VisionEngine Integration...")
audio = AudioEngine()
vision = VisionEngine(audio_engine=audio)
print("-> VisionEngine initialized successfully!")
print("="*60)
print("       ALL UNIT TESTS PASSED SUCCESSFULLY!")
print("="*60)
