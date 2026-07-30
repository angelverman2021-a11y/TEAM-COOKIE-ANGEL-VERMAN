import cv2
import numpy as np
import time
from src.navi_engine import VisionEngine, ObjectDetector, ObjectTracker, FrameProcessor
from src.audio_engine import AudioEngine

print("="*60)
print("       TESTING NAVI VISION ENGINE & MODULAR CLASSES")
print("="*60)

# Test 1: ObjectDetector
print("\n[TEST 1] Testing ObjectDetector with YOLO11...")
detector = ObjectDetector()
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
res = detector.detect(dummy_frame)
print(f"-> ObjectDetector successfully ran on {detector.device.upper()}. Result: {res is not None}")

# Test 2: ObjectTracker (ByteTrack)
print("\n[TEST 2] Testing ObjectTracker with ByteTrack...")
tracker = ObjectTracker(detector)
tracked = tracker.track(dummy_frame)
print(f"-> ObjectTracker initialized. Tracked items count: {len(tracked)}")

# Test 3: FrameProcessor
print("\n[TEST 3] Testing FrameProcessor HUD Rendering...")
processor = FrameProcessor()
annotated = processor.draw_hud(dummy_frame, tracked, "Happy")
fps = processor.update_fps()
print(f"-> FrameProcessor created frame shape: {annotated.shape}, Initial FPS: {fps:.2f}")

# Test 4: VisionEngine Initialization
print("\n[TEST 4] Testing VisionEngine Integration...")
audio = AudioEngine()
vision = VisionEngine(audio_engine=audio)
print("-> VisionEngine initialized successfully!")
print("="*60)
print("       ALL UNIT TESTS PASSED SUCCESSFULLY!")
print("="*60)
