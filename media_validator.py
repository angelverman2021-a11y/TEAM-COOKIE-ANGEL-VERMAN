import time
import numpy as np
import threading
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.navi_engine import VisionEngine
from src.audio_engine import AudioEngine
from src.vision.interfaces import PerceptionResult

print("="*50)
print("  MEDIA-BASED SYNTHETIC VALIDATION RUN")
print("="*50)

class MockAudio(AudioEngine):
    def __init__(self):
        super().__init__()
        self.log = []
        self._enabled = False # Don't actually speak
    def speak(self, text, priority=5):
        self.log.append((time.time(), priority, text))

class MockCameraManager:
    def __init__(self):
        self.frame_id = 0
        self.running = False
        
    def start(self):
        self.running = True
        
    def get_frame(self):
        if not self.running: return -1, None
        self.frame_id += 1
        # Generate dummy 640x480 frame
        return self.frame_id, np.zeros((480, 640, 3), dtype=np.uint8)
        
    def stop(self):
        self.running = False

# Mock providers to return deterministic scenario
class MockVisionProvider:
    def __init__(self):
        self.frame_count = 0
    def analyze(self, frame):
        self.frame_count += 1
        
        # Scenario: Person approaching from distance 0.1 to 0.9 over 30 frames
        dist = min(0.95, 0.1 + (self.frame_count * 0.03))
        
        # Box gets larger
        box_size = int(50 + dist * 200)
        
        return PerceptionResult(
            objects=[{"box": [100, 100, 100+box_size, 100+box_size], "label": "person"}],
            scene_status="Indoor Corridor",
            ocr_text="",
            navigation_context=""
        )

class MockDepthProvider:
    def __init__(self):
        self.frame_count = 0
    def estimate_depth(self, frame):
        self.frame_count += 1
        dist = min(0.95, 0.1 + (self.frame_count * 0.03))
        depth = np.zeros((480, 640), dtype=np.float32)
        depth[100:300, 100:300] = dist
        return depth

from unittest.mock import patch

with patch('src.navi_engine.FlorenceProvider', return_value=MockVisionProvider()), \
     patch('src.navi_engine.DepthAnythingProvider', return_value=MockDepthProvider()):
     
    # Setup
    audio = MockAudio()
    vision = VisionEngine(audio)
    vision.camera_manager = MockCameraManager()

    print("[VALIDATOR] Starting Media Playback Simulation...")
    vision.start()

    # Simulate 3 seconds of video processing
    time.sleep(3.0)

    vision.stop()

print("\n--- PERFORMANCE & UX METRICS ---")
fps = vision.diagnostics.get('cam_fps', 0)
latency = vision.diagnostics.get('ai_time', 0)

print(f"System FPS:         {fps:.2f} (Render Loop)")
print(f"AI Latency:         {latency:.2f} ms")
print(f"Total Warnings:     {len(audio.log)}")

print("\n--- AUDIO TIMELINE ---")
for t, pri, text in audio.log:
    print(f"[{pri}] {text}")

if any("collision" in t.lower() for _, _, t in audio.log):
    print("\nSUCCESS: Collision Avoidance trigger verified in synthetic video.")
else:
    print("\nFAILED: No collision detected.")
    sys.exit(1)
