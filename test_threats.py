import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.navigation_engine import CollisionPredictor, ThreatDetector

print("="*60)
print("       TESTING COLLISION & THREAT DETECTION")
print("="*60)

class MockAudio:
    def __init__(self):
        self.log = []
    def speak(self, text, priority):
        self.log.append((priority, text))

predictor = CollisionPredictor()
audio = MockAudio()
detector = ThreatDetector(audio)

now = time.time()

# Frame 1: Object far away
tracks = [{"track_id": 1, "label": "person", "dist": 0.4}]
col_data = predictor.update(tracks, now)
detector.check_threats(tracks, col_data)

# Frame 2: Object rapidly approaching 0.1s later
now += 0.1
tracks = [{"track_id": 1, "label": "person", "dist": 0.9}] # Dist jumped by 0.5 in 0.1s => velocity = 5.0
col_data = predictor.update(tracks, now)
detector.check_threats(tracks, col_data)

print(f"Collision Data: {col_data}")
print("Audio Log:")
for pri, text in audio.log:
    print(f" - [PRI {pri}]: {text}")

if any(pri == 1 for pri, text in audio.log) or any(pri == 2 for pri, text in audio.log):
    print("SUCCESS: Threat detected and prioritized correctly!")
else:
    print("FAILED: Did not detect fast/critical approach.")
    sys.exit(1)
