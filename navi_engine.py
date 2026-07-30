import cv2
from ultralytics import YOLO
import mediapipe as mp
import pickle
import collections
import warnings
warnings.filterwarnings("ignore") # Ignore scikit-learn warnings

def run_vision_engine(emotion_callback=None):
    print("=========================================")
    print("  Initializing Custom NAVI Engine...     ")
    print("=========================================")
    
    # 1. Load YOLO for Person Detection
    print("Loading YOLO Model...")
    yolo_model = YOLO('yolov8n.pt')

    # 2. Load our Custom Brain!
    print("Loading Custom ML Emotion Model...")
    try:
        with open("custom_emotion_model.pkl", 'rb') as f:
            emotion_model = pickle.load(f)
    except FileNotFoundError:
        print("CRITICAL ERROR: custom_emotion_model.pkl not found!")
        print("Please run train_model.py first.")
        return

    # 3. Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        min_detection_confidence=0.5
    )

    print("Opening Webcam...")
    camera = cv2.VideoCapture(0)

    # Smoothing Buffer
    current_emotion = "Scanning..."
    previous_emotion = None
    emotion_history = collections.deque(maxlen=10)

    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Run YOLO to find bodies
        results = yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        person_found = False
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:
                person_found = True
                break
                
        if person_found:
            # If we see a person, run our lightning-fast Custom Emotion Model
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(rgb_frame)
            
            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    nose_x = face_landmarks.landmark[1].x
                    nose_y = face_landmarks.landmark[1].y
                    
                    # --- CRITICAL ML FIX: SCALE NORMALIZATION ---
                    xs = [lm.x for lm in face_landmarks.landmark]
                    ys = [lm.y for lm in face_landmarks.landmark]
                    face_width = max(xs) - min(xs)
                    face_height = max(ys) - min(ys)
                    
                    row = []
                    for lm in face_landmarks.landmark:
                        row.append((lm.x - nose_x) / (face_width + 1e-6))
                        row.append((lm.y - nose_y) / (face_height + 1e-6))
                    
                    # Predict!
                    prediction = emotion_model.predict([row])[0]
                    emotion_history.append(prediction)
                    
                    # Temporal Smoothing
                    if len(emotion_history) > 0:
                        current_emotion = max(set(emotion_history), key=emotion_history.count)
            else:
                current_emotion = "Face not clearly visible"
        else:
            current_emotion = "No person detected"

        # Trigger callback for Audio
        if current_emotion != previous_emotion:
            if emotion_callback:
                emotion_callback(current_emotion)
            previous_emotion = current_emotion

        # UI Overlay
        cv2.rectangle(annotated_frame, (10, 10), (450, 70), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"NAVI MOOD: {current_emotion.upper()}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("NAVI Engine (Custom ML)", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision_engine()
