import cv2
import numpy as np
import threading
import collections
import warnings
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CAMERA_INDEX,
    CAMERA_RESOLUTION,
    YOLO_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    DEVICE,
    TARGET_FPS,
    SMOOTHING_BUFFER_SIZE
)

warnings.filterwarnings("ignore")


class ObjectDetector:
    """Class 1: ObjectDetector - Manages YOLO11 model loading and inference with CUDA/CPU support."""
    def __init__(self, model_path=YOLO_MODEL_PATH, device=DEVICE, conf_thresh=CONFIDENCE_THRESHOLD, iou_thresh=IOU_THRESHOLD):
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        print(f"[VISION-DETECTOR] Loading model '{model_path}' on device '{self.device}'...")
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        
        try:
            self.model.to(self.device)
            print(f"[VISION-DETECTOR] Model successfully assigned to {self.device.upper()}")
            # Warm up CUDA memory allocation
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model(dummy, conf=self.conf_thresh, verbose=False)
            print(f"[VISION-DETECTOR] CUDA warm-up complete!")
        except Exception as e:
            print(f"[VISION-DETECTOR] Device warning: {e}. Falling back to CPU.")
            self.device = "cpu"
            self.model.to("cpu")

    def detect(self, frame):
        """Runs object detection with confidence & NMS IOU thresholding."""
        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False
        )
        return results[0] if results else None


class ObjectTracker:
    """Class 2: ObjectTracker - Persistent object tracking across frames using pure Python IOU (prevents PyTorch CUDA deadlocks)."""
    def __init__(self, detector: ObjectDetector):
        self.detector = detector
        self.next_id = 1
        self.tracked_objects = {} # id -> {"box": [x1,y1,x2,y2], "conf": conf, "cls": cls_id, "label": label, "missed": 0}

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 < x1 or y2 < y1: return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / float(area1 + area2 - intersection)

    def track(self, frame):
        """Performs detection and purely python-based IOU tracking."""
        results = self.detector.model(
            frame,
            conf=self.detector.conf_thresh,
            iou=self.detector.iou_thresh,
            device=self.detector.device,
            verbose=False
        )
        
        current_detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = results[0].names[cls_id] if hasattr(results[0], 'names') else str(cls_id)
                current_detections.append({"box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])], "conf": conf, "cls": cls_id, "label": label})
        
        # Match detections to existing tracks
        new_tracked = {}
        for det in current_detections:
            best_id = None
            best_iou = 0.3
            for tid, tobj in self.tracked_objects.items():
                if tobj["cls"] != det["cls"]: continue
                iou = self._compute_iou(det["box"], tobj["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            
            if best_id is not None:
                new_tracked[best_id] = det
                new_tracked[best_id]["missed"] = 0
                del self.tracked_objects[best_id]
            else:
                new_tracked[self.next_id] = det
                new_tracked[self.next_id]["missed"] = 0
                self.next_id += 1
                
        # Keep missed objects for 5 frames
        for tid, tobj in self.tracked_objects.items():
            if tobj["missed"] < 5:
                tobj["missed"] += 1
                new_tracked[tid] = tobj

        self.tracked_objects = new_tracked
        
        out = []
        for tid, obj in self.tracked_objects.items():
            out.append({
                "box": obj["box"],
                "conf": obj["conf"],
                "cls": obj["cls"],
                "track_id": tid,
                "label": obj["label"]
            })
        return out



class FrameProcessor:
    """Class 3: FrameProcessor - Handles HUD overlays, bounding box drawing, and FPS counter."""
    def __init__(self):
        self.fps_buffer = collections.deque(maxlen=30)
        self.last_time = time.time()

    def update_fps(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps_buffer.append(1.0 / dt)
        return sum(self.fps_buffer) / len(self.fps_buffer) if len(self.fps_buffer) > 0 else 0.0

    def draw_hud(self, frame, tracked_objects, current_emotion):
        """Renders bounding boxes with ByteTrack IDs and HUD metrics cleanly onto the frame."""
        annotated = frame.copy()
        current_fps = self.update_fps()

        # Draw Tracked Bounding Boxes
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["box"]
            track_id = obj["track_id"]
            label = obj["label"]
            conf = obj["conf"]

            # Person gets pure neon green (BGR format: 0, 255, 100), other objects cyan (255, 200, 0)
            color = (0, 255, 100) if obj["cls"] == 0 else (255, 200, 0)
            
            # Bounding Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label & Track ID Tag
            id_str = f"ID:{track_id} " if track_id is not None else ""
            tag = f"{id_str}{label.upper()} {int(conf * 100)}%"
            
            # Label background box
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(th + 4, y1)), color, -1)
            cv2.putText(annotated, tag, (x1 + 3, max(th + 2, y1 - 4)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Dashboard Top Banner
        cv2.rectangle(annotated, (10, 10), (480, 75), (0, 0, 0), -1)
        cv2.putText(annotated, f"NAVI MOOD: {current_emotion.upper()}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        
        # FPS & Hardware Banner
        fps_color = (0, 255, 0) if current_fps >= 25 else (0, 255, 255)
        cv2.putText(annotated, f"FPS: {current_fps:.1f} | DEVICE: {DEVICE.upper()}", (20, 68), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1, cv2.LINE_AA)

        return annotated


class VisionEngine:
    """Class 4: VisionEngine - Main Orchestrator providing live high-FPS streaming and Flask integration."""
    def __init__(self, audio_engine):
        self.audio = audio_engine
        self.current_emotion = "Scanning..."
        self.previous_emotion = None
        self.emotion_history = collections.deque(maxlen=SMOOTHING_BUFFER_SIZE)
        self.running = False
        self.camera = None
        self.latest_jpeg_bytes = None
        self.is_analyzing_emotion = False
        self.frame_count = 0
        self.lock = threading.Lock()
        self.frame_ready_event = threading.Event()

        # Initialize AI Components
        self.detector = ObjectDetector(model_path=YOLO_MODEL_PATH, device=DEVICE)
        self.tracker = ObjectTracker(self.detector)
        self.processor = FrameProcessor()

        # DeepFace Warmup
        print("[VISION] Warming up DeepFace Engine...")
        try:
            from deepface import DeepFace
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False, detector_backend='skip')
        except Exception as e:
            print(f"[VISION] DeepFace warmup notice: {e}")

    def _analyze_emotion_async(self, person_crop, emotion_callback):
        """Asynchronous background worker for DeepFace emotion classification."""
        from deepface import DeepFace
        try:
            analysis = DeepFace.analyze(
                person_crop,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='skip'
            )
            if isinstance(analysis, list):
                emotion = analysis[0]['dominant_emotion']
            else:
                emotion = analysis['dominant_emotion']
                
            self.emotion_history.append(emotion)
            if len(self.emotion_history) > 0:
                self.current_emotion = max(set(self.emotion_history), key=self.emotion_history.count)
                
            if self.current_emotion != self.previous_emotion:
                if emotion_callback:
                    emotion_callback(self.current_emotion)
                self.previous_emotion = self.current_emotion
        except Exception:
            pass
        finally:
            self.is_analyzing_emotion = False

    def start(self, emotion_callback):
        """Starts main camera acquisition and rendering loop with retry tolerance."""
        print(f"[VISION] Opening Webcam (Trying indices {CAMERA_INDEX}, 1, 2, 0, 3, -1)...")
        
        self.camera = None
        for i in [CAMERA_INDEX, 1, 2, 0, 3, -1]:
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
                cam = cv2.VideoCapture(i, backend) if backend is not None else cv2.VideoCapture(i)
                if cam.isOpened():
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                    cam.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    for attempt in range(10):
                        success, frame = cam.read()
                        if success and frame is not None:
                            print(f"[VISION] Connected to camera index {i} on attempt {attempt+1}")
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
            
            error_frame = np.zeros((CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
            error_frame[:] = (0, 0, 150)
            cv2.putText(error_frame, "CAMERA DISCONNECTED", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            ret, buf = cv2.imencode('.jpg', error_frame)
            if ret:
                with self.lock:
                    self.latest_jpeg_bytes = buf.tobytes()
            self.audio.speak(msg)
            return

        self.running = True
        self.audio.speak("Vision system online. Streaming to dashboard.")

        consecutive_failures = 0

        while self.running:
            success, frame = self.camera.read()
            if not success or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 15:
                    msg = "Camera disconnected."
                    print(f"[ERROR] {msg}")
                    self.audio.speak(msg)
                    break
                time.sleep(0.01)
                continue
            
            consecutive_failures = 0
            self.frame_count += 1

            # High-speed CUDA Object Tracking
            try:
                tracked = self.tracker.track(frame)
            except Exception as e:
                tracked = []

            # Check if any person is detected
            person_objs = [obj for obj in tracked if obj["cls"] == 0]

            if person_objs:
                # Trigger async DeepFace analysis on largest person crop
                if not self.is_analyzing_emotion:
                    best_person = max(
                        person_objs, 
                        key=lambda o: (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1])
                    )
                    x1, y1, x2, y2 = best_person["box"]
                    h, w, _ = frame.shape
                    
                    # Isolate the Head/Face region (top 40% of person bounding box)
                    box_h = max(1, y2 - y1)
                    head_y2 = y1 + int(box_h * 0.40)
                    
                    face_crop = frame[max(0, y1):min(h, head_y2), max(0, x1):min(w, x2)]
                    
                    if face_crop.size > 0:
                        self.is_analyzing_emotion = True
                        threading.Thread(
                            target=self._analyze_emotion_async,
                            args=(face_crop.copy(), emotion_callback),
                            daemon=True
                        ).start()
            else:
                self.current_emotion = "No person detected"

            # Draw HUD Overlays
            annotated_frame = self.processor.draw_hud(frame, tracked, self.current_emotion)
            
            # Encode frame to JPEG safely inside lock with quality 85
            ret, buf = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ret:
                with self.lock:
                    self.latest_jpeg_bytes = buf.tobytes()
                self.frame_ready_event.set()

        self.camera.release()

    def get_frame(self):
        """Returns latest JPEG bytes for Flask MJPEG stream in a 100% thread-safe manner."""
        with self.lock:
            return self.latest_jpeg_bytes

    def stop(self):
        self.running = False
