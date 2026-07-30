import pyttsx3
import threading
import pythoncom
import queue
import time
import os
import sys

# Add root directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AUDIO_RATE, AUDIO_ENABLED

class AudioEngine:
    """
    Robust Audio Engine for Windows SAPI5 via pyttsx3.
    Supports Priority Queuing and Interruption for Emergency Threats.
    """
    def __init__(self):
        self.is_speaking = False
        self._enabled = AUDIO_ENABLED
        self.queue = queue.PriorityQueue()
        self.current_priority = 999
        self.interrupt_flag = False
        
        if self._enabled:
            self.worker_thread = threading.Thread(target=self._audio_loop, daemon=True)
            self.worker_thread.start()

    def speak(self, text, priority=5):
        """
        Non-blocking TTS trigger with priority.
        1 = Collision, 2 = Fast Object, 3 = Navigation, 4 = OCR, 5 = General
        """
        if not self._enabled:
            return
            
        # If higher priority comes in, interrupt the current speech immediately
        if priority < self.current_priority:
            self.interrupt_flag = True
            
        # Enqueue with priority and timestamp (to resolve ties via FIFO)
        self.queue.put((priority, time.time(), text))
        
    def _audio_loop(self):
        # CRITICAL: Windows COM objects must be initialized per thread.
        pythoncom.CoInitialize()
        engine = pyttsx3.init()
        engine.setProperty('rate', AUDIO_RATE)
        
        # Hook into the speech event loop to allow mid-sentence interruption
        def onWord(name, location, length):
            if self.interrupt_flag:
                engine.stop()
                
        engine.connect('started-word', onWord)
        
        while True:
            # Wait for next message
            priority, timestamp, text = self.queue.get()
            
            self.is_speaking = True
            self.current_priority = priority
            self.interrupt_flag = False
            
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[AUDIO ERROR]: {e}")
            finally:
                self.is_speaking = False
                self.current_priority = 999
