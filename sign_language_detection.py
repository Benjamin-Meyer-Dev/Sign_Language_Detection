import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from scipy import stats
from sklearn.model_selection import train_test_split

#===========================================================================================================================================

data_path = "Letter Data"
letters = np.array(['A', 'B', 'C'])
colors = [(245,117,16), (117,245,16), (16,117,245)]

num_sequences = 30
sequence_length = 30
start_folder = 30

model = Sequential()

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
    sequences, labels = [], []
    label_map = {label:num for num, label in enumerate(letters)}
    
    for letter in letters:
        for sequence in np.array(os.listdir(os.path.join(data_path, letter))).astype(int):
            window = []
            
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(data_path, letter, str(sequence), '{}.npy'.format(frame_num)))
                window.append(res)
                
            sequences.append(window)
            labels.append(label_map[letter])
            
    x = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    return train_test_split(x, y, test_size=0.05)
    
#===========================================================================================================================================

def build_model():
    x_train, x_test, y_train, y_test = train_model()
    
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(letters.shape[0], activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])
    model.save('action.h5')
    
    letter_detection()
    
#===========================================================================================================================================

def prob_viz(res, input_frame):
    output_frame = input_frame.copy()
    
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, letters[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

#===========================================================================================================================================

def letter_detection():
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    
    model = tf.keras.models.load_model('action.h5')

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(np.argmax(res))
                predictions.append(np.argmax(res))
                
                if np.unique(predictions[-10:])[0] == np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if letters[np.argmax(res)] != sentence[-1]:
                                sentence.append(letters[np.argmax(res)])
                        else:
                            sentence.append(letters[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                image = prob_viz(res, image)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('OpenCV Feed', image)

            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q') or cv2.getWindowProperty('Sign Language Detection', cv2.WND_PROP_VISIBLE) < 1:
                break
            
        cap.release()
        cv2.destroyAllWindows()

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
        cv2.putText(frame, '3) Train Model', (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, '4) Letter Detection', (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Sign Language Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('1'):
            camera_detection()
            break
        elif key == ord('2'):
            collect_data()
            break
        elif key == ord('3'):
            build_model()
            break
        elif key == ord('4'):
            letter_detection()
            break
        elif cv2.getWindowProperty('Sign Language Detection', cv2.WND_PROP_VISIBLE) < 1:
            break
        
    cv2.destroyAllWindows()

#===========================================================================================================================================