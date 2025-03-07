import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import tensorflow as tf

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

#===========================================================================================================================================

data_path = "Letter Data"
letters = np.array(['A', 'B', 'C'])
num_sequences = 30
sequence_length = 30
start_folder = 30

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

#===========================================================================================================================================

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False 
    
    results = model.process(image)
    
    image.flags.writeable = True 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results

#===========================================================================================================================================

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 

#===========================================================================================================================================

def camera_detection():
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            cv2.imshow('Sign Language Detection', image)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q') or cv2.getWindowProperty('Sign Language Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
            
        cap.release()
        cv2.destroyAllWindows()

#===========================================================================================================================================

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

#===========================================================================================================================================

def collect_data():
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for letter in letters:
            for sequence in range(num_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)
                    
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(letter, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('Sign Language Detection', image)
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(letter, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('Sign Language Detection', image)
                    
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(data_path, letter, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    key = cv2.waitKey(10) & 0xFF
                    
                    if key == ord('q') or cv2.getWindowProperty('Sign Language Detection', cv2.WND_PROP_VISIBLE) < 1:
                        break
                        
        cap.release()
        cv2.destroyAllWindows()

#===========================================================================================================================================

def train_model():
    label_map = {label:num for num, label in enumerate(letters)}
    print(label_map)

#===========================================================================================================================================

if __name__ == "__main__":
    for letter in letters: 
        for sequence in range(num_sequences):
            try: 
                os.makedirs(os.path.join(data_path, letter, str(sequence)))
            except:
                pass
            
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        cv2.putText(frame, 'Select mode:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, '1) Point Tracking', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, '2) Data Collection', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Sign Language Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('1'):
            camera_detection()
            break
        elif key == ord('2'):
            collect_data()
            break
        elif cv2.getWindowProperty('Sign Language Detection', cv2.WND_PROP_VISIBLE) < 1:
            break
        
    cv2.destroyAllWindows()

#===========================================================================================================================================