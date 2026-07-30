import cv2
from ultralytics import YOLO
from deepface import DeepFace
import threading
import collections
import warnings
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAMERA_INDEX, YOLO_MODEL_PATH, SMOOTHING_BUFFER_SIZE

warnings.filterwarnings("ignore")

class VisionEngine:
    def __init__(self, audio_engine):
        self.audio = audio_engine
        self.current_emotion = "Scanning..."
        self.previous_emotion = None
        self.emotion_history = collections.deque(maxlen=SMOOTHING_BUFFER_SIZE)
        self.running = False
        self.camera = None
        self.latest_frame = None
        self.is_analyzing = False
        self.frame_count = 0
        
        print("[VISION] Initializing YOLO...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        
        print("[VISION] Initializing DeepFace Engine...")
        # We pre-load the model here so it doesn't stutter on the first frame
        try:
            # Dummy analysis to warm up the model
            import numpy as np
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        except Exception as e:
            print(f"[VISION] DeepFace warmup notice: {e}")

    def start(self, emotion_callback):
        print(f"[VISION] Opening Webcam (Trying indices 0, 1, 2)...")
        
        # Try multiple camera indices with warm-up retries
        self.camera = None
        import time
        for i in [CAMERA_INDEX, 1, 2, 0]:
            # Try DSHOW first, then default MSMF fallback
            for backend in [cv2.CAP_DSHOW, None]:
                cam = cv2.VideoCapture(i, backend) if backend is not None else cv2.VideoCapture(i)
                if cam.isOpened():
                    # Warm-up retry loop: Webcams often take ~100-300ms to output the first valid frame
                    for attempt in range(8):
                        success, frame = cam.read()
                        if success and frame is not None:
                            print(f"[VISION] Successfully connected to camera index {i} (attempt {attempt+1})")
                            self.camera = cam
                            break
                        time.sleep(0.1)
                    if self.camera is not None:
                        break
                cam.release()
            if self.camera is not None:
                break
            
        if self.camera is None or not self.camera.isOpened():
            msg = "Error. Could not connect to camera. Please check your webcam."
            print(f"[ERROR] {msg}")
            self.current_emotion = "CAMERA FAILED"
            
            import numpy as np
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            error_frame[:] = (0, 0, 150) # Dark red background
            cv2.putText(error_frame, "CAMERA DISCONNECTED", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            self.latest_frame = error_frame
            
            self.audio.speak(msg)
            return

        self.running = True
        self.audio.speak("Vision system online. Streaming to dashboard.")

        while self.running:
            success, frame = self.camera.read()
            if not success:
                msg = "Camera disconnected."
                print(f"[ERROR] {msg}")
                self.audio.speak(msg)
                break
                
            self.frame_count += 1
            self._process_frame(frame, emotion_callback)
                
        self.camera.release()
        
    def _analyze_emotion_in_background(self, frame_to_analyze):
        try:
            # RetinaFace is highly accurate for face cropping.
            analysis = DeepFace.analyze(
                frame_to_analyze, 
                actions=['emotion'], 
                enforce_detection=False,
                detector_backend='retinaface'
            )
            if isinstance(analysis, list):
                emotion = analysis[0]['dominant_emotion']
            else:
                emotion = analysis['dominant_emotion']
                
            self.emotion_history.append(emotion)
            
            if len(self.emotion_history) > 0:
                self.current_emotion = max(set(self.emotion_history), key=self.emotion_history.count)
                
        except Exception as e:
            pass
        finally:
            self.is_analyzing = False
            
    def _process_frame(self, frame, emotion_callback):
        try:
            # Run YOLO to find bodies
            results = self.yolo_model(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            person_found = any(int(box.cls[0]) == 0 for box in results[0].boxes)
                    
            if person_found:
                # Run DeepFace in a background thread every 10 frames to avoid lagging the video
                if self.frame_count % 10 == 0 and not self.is_analyzing:
                    self.is_analyzing = True
                    frame_copy = frame.copy()
                    thread = threading.Thread(target=self._analyze_emotion_in_background, args=(frame_copy,))
                    thread.daemon = True
                    thread.start()
            else:
                self.current_emotion = "No person detected"

            # Trigger callback
            if self.current_emotion != self.previous_emotion:
                if emotion_callback:
                    emotion_callback(self.current_emotion)
                self.previous_emotion = self.current_emotion

            # UI Overlay for Web Stream
            cv2.rectangle(annotated_frame, (10, 10), (450, 70), (0, 0, 0), -1)
            cv2.putText(annotated_frame, f"NAVI MOOD: {self.current_emotion.upper()}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Store latest frame for Flask streaming instead of cv2.imshow
            self.latest_frame = annotated_frame
        except Exception as e:
            import traceback
            print(f"[VISION PROCESS_FRAME ERROR] {e}")
            traceback.print_exc()
        
    def get_frame(self):
        """Encodes the latest frame as JPEG for web streaming."""
        if self.latest_frame is None:
            return None
        ret, buffer = cv2.imencode('.jpg', self.latest_frame)
        return buffer.tobytes()
        
    def stop(self):
        self.running = False
