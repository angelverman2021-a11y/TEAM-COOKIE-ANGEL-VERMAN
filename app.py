import streamlit as st
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import threading

# 1. FIX APP LOAD TIME: Cache the YOLO model so it only loads once!
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

model = load_yolo_model()

st.set_page_config(page_title="NAVI", layout="wide")
st.title("👁️ NAVI")
st.subheader("Navigation Assistant for the Visually Impaired")

frame_window = st.image([])
run_camera = st.checkbox("Turn on Camera")

# Global variables for threading
current_emotion = "Scanning for faces..."
is_analyzing = False

def analyze_emotion_in_background(frame_to_analyze):
    global current_emotion, is_analyzing
    try:
        # 2. FIX ACCURACY: Switch back to 'retinaface'. 
        # Since it is in a background thread now, it won't lag the camera!
        # RetinaFace is much more accurate at cropping faces for the emotion model.
        analysis = DeepFace.analyze(
            frame_to_analyze, 
            actions=['emotion'], 
            enforce_detection=False,
            detector_backend='retinaface'
        )
        if isinstance(analysis, list):
            current_emotion = analysis[0]['dominant_emotion']
        else:
            current_emotion = analysis['dominant_emotion']
    except Exception as e:
        # If it misses a frame, just silently pass to avoid flickering text
        pass
    finally:
        is_analyzing = False

if run_camera:
    # 3. FIX CAMERA BOOT TIME: Only open the webcam if the box is checked
    camera = cv2.VideoCapture(0)
    frame_count = 0
    
    while run_camera:
        success, frame = camera.read()
        if not success:
            st.error("Failed to capture video.")
            break
        
        frame_count += 1
        
        # Run YOLO Person Detection
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # Run Emotion AI in Background Thread
        # We can safely run this more often (every 15 frames) now that it is threaded!
        if frame_count % 15 == 0 and not is_analyzing:
            is_analyzing = True
            frame_copy = frame.copy()
            thread = threading.Thread(target=analyze_emotion_in_background, args=(frame_copy,))
            thread.start()
                
        # Draw the text
        cv2.putText(annotated_frame, f"Mood: {current_emotion.upper()}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
        final_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_window.image(final_frame)
        
    camera.release()
