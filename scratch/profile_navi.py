import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.navi_engine import VisionEngine
from src.audio_engine import AudioEngine

print("="*40)
print("  NAVI ENGINE PROFILER (PHASE E)")
print("="*40)

audio = AudioEngine()
vision = VisionEngine(audio)

t0 = time.time()
print("[PROFILER] Booting VisionEngine...")
vision.start()
t1 = time.time()

print(f"[PROFILER] Boot time: {t1 - t0:.2f} seconds")
print("[PROFILER] Waiting for models to warm up...")

# Wait until active
for i in range(30):
    if len(vision.diagnostics) > 0 and vision.diagnostics.get("cam_fps", 0) > 0:
        break
    time.sleep(1.0)

t2 = time.time()
print(f"[PROFILER] Models warmed up in {t2 - t1:.2f} seconds.")

print("[PROFILER] Profiling for 10 seconds...")
start_profile = time.time()
samples = []
while time.time() - start_profile < 10.0:
    samples.append(vision.diagnostics.copy())
    time.sleep(1.0)

vision.stop()

# Aggregate results
valid_samples = [s for s in samples if "ai_time" in s]
if len(valid_samples) == 0:
    print("[PROFILER] No valid samples collected.")
    sys.exit(1)

avg_cam_fps = sum(s.get("cam_fps", 0) for s in valid_samples) / len(valid_samples)
avg_ai_time = sum(s.get("ai_time", 0) for s in valid_samples) / len(valid_samples)

print("\n--- RESULTS ---")
print(f"Average Camera FPS:  {avg_cam_fps:.2f} FPS")
print(f"Average AI Latency:  {avg_ai_time:.2f} ms")
print(f"Approx AI FPS:       {1000.0 / avg_ai_time if avg_ai_time > 0 else 0:.2f} FPS")
print(f"Device Used:         {valid_samples[-1].get('device')}")

print("\n[PROFILER] Done.")
