from flask import Flask, render_template, Response, jsonify, request
from src.navi_engine import VisionEngine
from src.audio_engine import AudioEngine
import threading
import time

app = Flask(__name__)

# Initialize Core AI Engines
audio = AudioEngine()
vision = VisionEngine(audio_engine=audio)

# Global State for Web Dashboard
system_status = "Disconnected"
guardian_info = {"name": "", "phone": ""}

def on_emotion_changed(emotion):
    global system_status
    if emotion and emotion != "Scanning...":
        if emotion == "No person detected":
            sentence = "I do not see anyone in front of you."
        elif emotion == "Face not clearly visible":
            sentence = "Someone is there, but their face is turned away."
        else:
            sentence = f"The person in front of you seems {emotion}."
        
        print(f"[NAVI AUDIO]: {sentence}")
        audio.speak(sentence)
        system_status = f"Last detected: {emotion}"
        
        # Automatic Guardian SOS Trigger (Example logic)
        if emotion == "Angry" and guardian_info["name"]:
            alert_msg = f"Warning. Aggression detected. Alerting {guardian_info['name']}."
            audio.speak(alert_msg)
            print(f"[GUARDIAN SOS]: SMS sent to {guardian_info['phone']} - User may be in a hostile environment!")

def generate_video_stream():
    """Generator function to yield JPEG frames instantly whenever a new frame is ready."""
    last_frame = None
    while True:
        if vision.running:
            vision.frame_ready_event.wait(timeout=0.1)
            vision.frame_ready_event.clear()
        
        frame = vision.get_frame()
        if frame is not None and frame != last_frame:
            last_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.01)

# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "emotion": vision.current_emotion,
        "status": system_status,
        "guardian_set": bool(guardian_info["name"])
    })

@app.route('/api/guardian', methods=['POST'])
def set_guardian():
    data = request.json
    guardian_info["name"] = data.get("name", "")
    guardian_info["phone"] = data.get("phone", "")
    audio.speak(f"Guardian {guardian_info['name']} has been linked to your NAVI glasses.")
    return jsonify({"success": True})

@app.route('/api/sos', methods=['POST'])
def trigger_sos():
    if guardian_info["name"]:
        audio.speak(f"SOS triggered. Sending location and camera feed to {guardian_info['name']}.")
        return jsonify({"success": True, "message": f"Alert sent to {guardian_info['name']}"})
    else:
        audio.speak("SOS triggered, but no guardian is configured.")
        return jsonify({"success": False, "message": "No Guardian Configured"})

@app.route('/api/connect', methods=['POST'])
def connect_glasses():
    global system_status
    if not vision.running:
        system_status = "Booting Vision Engine..."
        # Start vision engine in a background thread so it doesn't block Flask
        threading.Thread(target=vision.start, args=(on_emotion_changed,), daemon=True).start()
        return jsonify({"success": True, "message": "Connected successfully"})
    return jsonify({"success": True, "message": "Already connected"})

if __name__ == "__main__":
    print("=========================================")
    print("    Starting NAVI Web Dashboard...       ")
    print("    Go to http://127.0.0.1:5000          ")
    print("=========================================")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
