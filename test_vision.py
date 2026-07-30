import time
from src.navi_engine import VisionEngine
from src.audio_engine import AudioEngine
import threading

audio = AudioEngine()
vision = VisionEngine(audio_engine=audio)

def dummy_callback(emotion):
    pass

print("Starting vision engine synchronously...")
try:
    vision.start(dummy_callback)
except Exception as e:
    import traceback
    traceback.print_exc()

print("Checking if thread is still running:", vision.running)
