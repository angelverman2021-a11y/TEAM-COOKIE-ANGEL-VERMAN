import os
import cv2
import mediapipe as mp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import skops.io as sio
import warnings

from config import CUSTOM_MODEL_PATH

warnings.filterwarnings("ignore")

DATA_DIR = r"C:\Users\Pranjal\NAVI\emotion folder"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

data = []

print("=========================================")
print("  NAVI Secure Auto-Trainer 🚀            ")
print("=========================================")
print("Scanning image folders...")

for emotion_name in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, emotion_name)
    if os.path.isdir(folder_path):
        print(f"Processing folder: {emotion_name.upper()}...")
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_img)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    nose_x = face_landmarks.landmark[1].x
                    nose_y = face_landmarks.landmark[1].y
                    
                    xs = [lm.x for lm in face_landmarks.landmark]
                    ys = [lm.y for lm in face_landmarks.landmark]
                    face_width = max(xs) - min(xs)
                    face_height = max(ys) - min(ys)
                    
                    row = [emotion_name.capitalize()]
                    for lm in face_landmarks.landmark:
                        row.append((lm.x - nose_x) / (face_width + 1e-6))
                        row.append((lm.y - nose_y) / (face_height + 1e-6))
                    
                    data.append(row)
            else:
                print(f"  Warning: MediaPipe couldn't see a face clearly in {filename}")

if len(data) == 0:
    print("\nCRITICAL ERROR: No faces were detected in any of your images!")
    exit()

columns = ['label'] + [f'val_{i}' for i in range(468 * 2)]
df = pd.DataFrame(data, columns=columns)

X = df.drop('label', axis=1)
y = df['label']

print("Training Secure Machine Learning Model (Random Forest)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

print(f"\n=========================================")
print(f"  MODEL ACCURACY: {accuracy * 100:.2f}%")
print(f"=========================================\n")

# SECURE SERIALIZATION: Using skops instead of pickle closes the RCE vulnerability.
sio.dump(model, CUSTOM_MODEL_PATH)
    
print(f"Success! Secure model saved to: {CUSTOM_MODEL_PATH}")
