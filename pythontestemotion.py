import cv2
import numpy as np
import pyttsx3
import threading
from tensorflow.keras.models import model_from_json

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Select English voice
voices = engine.getProperty('voices')
for voice in voices:
    if "english" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

# Emotion labels
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# Load model
try:
    with open('model/emotion_model.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights("model/emotion_model.h5")

    print("✅ Emotion model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Load face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("📱 Starting MobileNetV2-Based Facial Emotion Recognition...")

prev_emotion = None

# Function to speak emotion
def speak_emotion(emotion):
    engine.say(f"You are feeling {emotion}")
    engine.runAndWait()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture image.")
        break

    # Add heading on screen
    cv2.putText(frame, "MobileNetV2-Based Facial Emotion Recognition",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10),
                      (0, 255, 0), 3)

        # Preprocess face
        roi_gray = gray_frame[y:y + h, x:x + w]
        resized = cv2.resize(roi_gray, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))

        # Predict emotion
        prediction = emotion_model.predict(reshaped)
        maxindex = int(np.argmax(prediction))
        detected_emotion = emotion_dict[maxindex]

        # Display emotion
        cv2.putText(frame, detected_emotion, (x + 10, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)

        # Speak only if emotion changes
        if detected_emotion != prev_emotion:
            threading.Thread(target=speak_emotion,
                             args=(detected_emotion,)).start()
            prev_emotion = detected_emotion

    # Show window
    cv2.imshow('Emotion Detection', frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()