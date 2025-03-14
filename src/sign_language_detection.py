import cv2
import os
import string
import sys

import mediapipe as mp
import numpy as np
import tensorflow as tf

from gui import GUI

from PyQt5.QtWidgets import QApplication

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

#=============================================================================================================================================

camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

DATASET_DIR = "ASL Dataset"
MODEL_DIR = "ASL AI Model"
LETTER_EXAMPLES_DIR = "Letter Examples"
LETTERS = {i: letter for i, letter in enumerate(string.ascii_uppercase[:5])}

for letter in LETTERS.values():
    os.makedirs(os.path.join(DATASET_DIR, letter), exist_ok=True)

capture_counts = {letter: len(os.listdir(os.path.join(DATASET_DIR, letter))) for letter in LETTERS.values()}

#=============================================================================================================================================

def collect_data():
    x, y = [], []
    
    print("Press a letter to collect data. Press 'Spacebar' to quit.")
    
    while camera.isOpened():
        ret, frame = camera.read()
        
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
                cv2.putText(frame, "Press a letter to collect data.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key in [ord('a'), ord('b'), ord('c'), ord('d'), ord('e')]:
                    action = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}[chr(key)]
                    
                    x.append(landmarks)
                    y.append(action)
                    
                    action_dir = os.path.join(DATASET_DIR, LETTERS[action])
                    frame_filename = f"{len(os.listdir(action_dir)) + 1}.npy"
                    
                    np.save(os.path.join(action_dir, frame_filename), landmarks)
                    
                    print(f"Captured {LETTERS[action]}: {frame_filename}")

        cv2.imshow("Sign Language Data Collection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord(' ') or cv2.getWindowProperty("Sign Language Data Collection", cv2.WND_PROP_VISIBLE) < 1:
            break
    
    camera.release()
    cv2.destroyAllWindows()

#=============================================================================================================================================

def load_dataset():
    x, y = [], []
    
    for label, letter in LETTERS.items():
        letter_dir = os.path.join(DATASET_DIR, letter)
        
        for file in os.listdir(letter_dir):
            file_path = os.path.join(letter_dir, file)
            x.append(np.load(file_path))
            y.append(label)
            
    return np.array(x), np.array(y)

#=============================================================================================================================================

def create_model():
    model = Sequential([
        Flatten(input_shape=(21, 3)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(LETTERS), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

#=============================================================================================================================================

def train_model():
    if not os.path.exists(DATASET_DIR):
        print("No dataset found! Please collect data first.")
        return
    
    x, y = load_dataset()
    
    if x.size == 0:
        print("No data available!")
        return
    
    model = create_model()
    model.fit(x, y, epochs=2000, validation_split=0.2)
    model.save(os.path.join(MODEL_DIR, "asl_model.h5"))
    
    print("Model trained and saved!")

#=============================================================================================================================================

def live_detection():
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "asl_model.h5")) if os.path.exists(os.path.join(MODEL_DIR, "asl_model.h5")) else None
    
    if model is None:
        print("No trained model found! Train the model first.")
        return
        
    while camera.isOpened():
        ret, frame = camera.read()
        
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).reshape(1, 21, 3)
                
                prediction = model.predict(landmarks)
                action = LETTERS[np.argmax(prediction)]
                
                cv2.putText(frame, action, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        cv2.imshow("Sign Language Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord(' ') or cv2.getWindowProperty("Sign Language Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    camera.release()
    cv2.destroyAllWindows()

#=============================================================================================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(app.exec_())

#=============================================================================================================================================
