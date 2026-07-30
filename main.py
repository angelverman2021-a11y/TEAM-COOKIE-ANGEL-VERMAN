from navi_engine import run_vision_engine
from audio_engine import AudioEngine

def main():
    print("=========================================")
    print("    Starting NAVI Orchestrator...        ")
    print("=========================================")
    
    # Initialize the Voice
    audio = AudioEngine()
    audio.speak("NAVI System Initialized. Scanning environment.")
    
    # Define what happens when the Vision Engine sees a new emotion
    def on_emotion_changed(emotion):
        if emotion and emotion != "Scanning for faces...":
            if emotion == "No person detected":
                sentence = "I do not see anyone in front of you."
            else:
                sentence = f"The person in front of you seems {emotion}."
            
            print(f"[NAVI AUDIO]: {sentence}")
            audio.speak(sentence)
            
    # Start the Vision Engine and pass it our audio rule
    print("Booting Vision System...")
    run_vision_engine(emotion_callback=on_emotion_changed)

if __name__ == "__main__":
    main()
