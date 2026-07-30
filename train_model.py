import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

CSV_FILE = "emotion_dataset.csv"
MODEL_FILE = "custom_emotion_model.pkl"

if not os.path.exists(CSV_FILE):
    print(f"Error: {CSV_FILE} not found! Please run data_collection.py first to gather data.")
    exit()

print("Loading dataset...")
df = pd.read_csv(CSV_FILE)

if len(df) < 50:
    print("Warning: You have very little data. The model might not be accurate.")
    print(f"Current rows: {len(df)}. Recommended: 500+")

# Split features (X) and labels (y)
X = df.drop('label', axis=1)
y = df['label']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Custom Random Forest Model on {len(X_train)} samples...")
# Initialize the model (Random Forest is extremely robust for this type of tabular data)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train it!
model.fit(X_train, y_train)

print("Training Complete. Evaluating...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=========================================")
print(f"  MODEL ACCURACY: {accuracy * 100:.2f}%")
print(f"=========================================\n")
print("Detailed Report:")
print(classification_report(y_test, y_pred))

# Save the model to disk so our engine can use it
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)
    
print(f"Success! Model saved to {MODEL_FILE}")
print("You can now update navi_engine.py to use this model!")
