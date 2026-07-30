import pyttsx3
import threading
import pythoncom

class AudioEngine:
    def __init__(self):
        # Flag to prevent NAVI from talking over herself
        self.is_speaking = False

    def _speak_thread(self, text):
        self.is_speaking = True
        
        # CRITICAL FIX FOR WINDOWS: 
        # Background threads need to initialize COM objects to use the Windows speech API.
        pythoncom.CoInitialize()
        
        engine = pyttsx3.init()
        engine.setProperty('rate', 160) # Slower, clearer speech
        
        engine.say(text)
        engine.runAndWait()
        
        self.is_speaking = False

    def speak(self, text):
        """
        Speaks the given text out loud. 
        Uses threading so the camera feed doesn't freeze while talking.
        """
        if not self.is_speaking:
            thread = threading.Thread(target=self._speak_thread, args=(text,))
            thread.daemon = True
            thread.start()
