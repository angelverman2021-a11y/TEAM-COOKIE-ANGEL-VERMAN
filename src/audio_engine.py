import pyttsx3
import threading
import pythoncom
import os
import sys

# Add root directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AUDIO_RATE, AUDIO_ENABLED

class AudioEngine:
    """
    Robust Audio Engine for Windows SAPI5 via pyttsx3.
    """
    def __init__(self):
        self.is_speaking = False
        self._enabled = AUDIO_ENABLED

    def _speak_thread(self, text):
        self.is_speaking = True
        try:
            # CRITICAL: Windows COM objects must be initialized per thread.
            pythoncom.CoInitialize()
            engine = pyttsx3.init()
            engine.setProperty('rate', AUDIO_RATE)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[AUDIO ERROR]: {e}")
        finally:
            self.is_speaking = False

    def speak(self, text):
        """
        Non-blocking TTS trigger.
        """
        if not self._enabled:
            return
            
        if not self.is_speaking:
            thread = threading.Thread(target=self._speak_thread, args=(text,))
            thread.daemon = True
            thread.start()
