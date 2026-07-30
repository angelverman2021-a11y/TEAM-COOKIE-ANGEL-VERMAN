import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)

CSV_FILE = "emotion_dataset.csv"

# Classes we want to train
# 0: Neutral, 1: Happy, 2: Angry, 3: Sad, 4: Surprised
EMOTIONS = {
    '0': 'Neutral',
    '1': 'Happy',
    '2': 'Angry',
    '3': 'Sad',
    '4': 'Surprised'
}

# Create the CSV file and headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        # 468 landmarks * 2 (x,y) = 936 columns, plus 1 for the label
        headers = ['label'] + [f'val_{i}' for i in range(468 * 2)]
        writer.writerow(headers)

print("=========================================")
print("  NAVI Custom ML Data Collector 🚀       ")
print("=========================================")
print("Instructions:")
print("- Make a face for the specific emotion.")
print("- Press the corresponding number key to record that frame!")
print("- Hold the key down while moving your head slightly to gather more data.")
for k, v in EMOTIONS.items():
    print(f"  Press {k} for {v}")
print("Press 'q' to quit.")
print("=========================================")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Draw the face mesh
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw dots on the face for visual feedback
            for lm in face_landmarks.landmark:
                h, w, _ = frame.shape
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

            # Get key press
            key = cv2.waitKey(1) & 0xFF
            char_key = chr(key) if key < 256 else ''

            if char_key in EMOTIONS.keys():
                # We pressed a valid key! Let's extract the geometry
                row = [EMOTIONS[char_key]]
                
                # To make it rotation/position invariant, we calculate relative to the nose (landmark 1)
                nose_x = face_landmarks.landmark[1].x
                nose_y = face_landmarks.landmark[1].y
                
                for lm in face_landmarks.landmark:
                    # Save relative X and Y positions
                    row.append(lm.x - nose_x)
                    row.append(lm.y - nose_y)
                
                # Append to CSV
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                    
                print(f"Recorded: {EMOTIONS[char_key]}")

            elif char_key == 'q':
                cap.release()
                cv2.destroyAllWindows()
                exit()

    # Show instructions on screen
    cv2.putText(frame, "Press 0:Neutral, 1:Happy, 2:Angry, 3:Sad, 4:Surprised", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow('Data Collector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
