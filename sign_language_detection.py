import cv2
import mediapipe as mp
import numpy as np
import os
import time

from matplotlib import pyplot as plt

#==============================================================================================================================

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    results = model.process(image)
    image.flags.writeable = True
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results

#==============================================================================================================================

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
#==============================================================================================================================
    
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

#==============================================================================================================================

def extract_keypoints(results):
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    
    return np.concatenate([left_hand, right_hand])

#==============================================================================================================================

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'test'])
no_sequences = 30
sequence_length = 30
start_folder = 30

for action in actions:    
    for sequence in range(1, no_sequences + 1):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while camera.isOpened():
        ret, frame = camera.read()
        
        image, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        
        if ret:
            action = 'hello'
            sequence = 1
            folder_path = os.path.join(DATA_PATH, action, str(sequence))
            
            np.save(os.path.join(folder_path, str(time.time())), keypoints)
        
        draw_styled_landmarks(image, results)
        cv2.imshow('Open CV Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    camera.release()
    cv2.destroyAllWindows()