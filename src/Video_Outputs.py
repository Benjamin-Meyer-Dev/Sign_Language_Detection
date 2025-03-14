import cv2
import Constants
import json
import os
import random
import sqlite3
import time

import numpy as np
import tensorflow as tf

from scipy.interpolate import interp1d
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

#=============================================================================================================================================

#Function for live sign language detection
def liveDetection(camera, hands, mpDrawing, mpHands, updateFrame, stopEvent):
    model = tf.keras.models.load_model(Constants.MODEL_PATH) if os.path.exists(Constants.MODEL_PATH) else None
    
    if model is None:
        print("No trained model found! Train the model first.")
        return
        
    while camera.isOpened():
        if stopEvent.is_set():
            break
        
        ret, frame = camera.read()
        
        if not ret:
            break
        
        frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRgb)

        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mpDrawing.draw_landmarks(frame, handLandmarks, mpHands.HAND_CONNECTIONS)
                
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in handLandmarks.landmark])
                landmarksArray = np.tile(landmarks, (Constants.COLLECTION_LENGTH, 1, 1))
                
                prediction = model.predict(landmarksArray.reshape(1, Constants.COLLECTION_LENGTH, Constants.HAND_POINTS, Constants.COORD_POINTS))
                action = Constants.LETTERS[np.argmax(prediction)]
                
                cv2.putText(frame, action, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
        updateFrame(frame)

#=============================================================================================================================================

#Function to collect data for training
def dataCollection(camera, hands, mpDrawing, mpHands, updateFrame, getKeyPress, resetKey, stopEvent):    
    landmarksCollection = []
    collecting = False
    letter = None
    collectionNumber = 0
    
    while camera.isOpened():
        if stopEvent.is_set():
            break
        
        ret, frame = camera.read()
        
        if not ret:
            break
        
        frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRgb)
        
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mpDrawing.draw_landmarks(frame, handLandmarks, mpHands.HAND_CONNECTIONS)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in handLandmarks.landmark])
                
                if collecting:
                    cv2.putText(frame, "Collecting data...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                else:
                    cv2.putText(frame, "Press a letter to collect data.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    
                letter = getKeyPress()
                
                if letter and not collecting:
                    print(f"Start collecting data for: {letter}")
                    
                    landmarksCollection = []
                    collecting = True
                    
                    startTime = time.time()
                    endTime = startTime + Constants.COLLECTION_LENGTH
                    
                    collectionNumber = getCollectionCount(letter) + 1
                    
                    frameCount = 0
                    interval = Constants.COLLECTION_LENGTH / Constants.COLLECTION_SNAPSHOTS
                    
                    nextCaptureTime = startTime
                    
                if collecting:
                    currentTime = time.time()
                    
                    if currentTime >= nextCaptureTime:
                        landmarksCollection.append(landmarks.tolist())
                        frameCount += 1
                        nextCaptureTime += interval
                        
                    if currentTime >= endTime or frameCount >= Constants.COLLECTION_SNAPSHOTS:
                        collecting = False
                        
                        if len(landmarksCollection) < Constants.COLLECTION_SNAPSHOTS:
                            landmarksCollection = interpolateSnapshots(landmarksCollection)
                        elif len(landmarksCollection) > Constants.COLLECTION_SNAPSHOTS:
                            landmarksCollection = downsampleSnapshots(landmarksCollection)
                            
                        insertLandmarks(letter, landmarksCollection)
                        resetKey()
                        
                        print(f"Captured {letter} collection {collectionNumber}: {len(landmarksCollection)} snapshots stored in SQLite.")
                        
        updateFrame(frame)
        
#=============================================================================================================================================

#Function for normal video display
def noDetection(camera, updateFrame, stopEvent):
    while camera.isOpened():
        if stopEvent.is_set():
            break
        
        ret, frame = camera.read()
        
        if not ret:
            break
        
        updateFrame(frame)

#=============================================================================================================================================

#Function to access the database for collection count
def getCollectionCount(letter):
    conn = sqlite3.connect(Constants.DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {letter}")
        result = cursor.fetchone()[0]
    finally:
        conn.close()

    return result

#=============================================================================================================================================

#Function to get the landmark data from the database
def getLandmarks():
    conn = sqlite3.connect(Constants.DATABASE_PATH)
    cursor = conn.cursor()
    
    landmarkData = {}
    
    for letter in Constants.LETTERS.values():
        cursor.execute(f"SELECT landmarks FROM {letter}")
        rows = cursor.fetchall()
        landmarkData[letter] = [json.loads(row[0]) for row in rows]
        
    conn.close()
    
    return landmarkData

#=============================================================================================================================================

#Fucntion to insert landmark data into the database
def insertLandmarks(letter, landmarksCollection):
    conn = sqlite3.connect(Constants.DATABASE_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            f"INSERT INTO {letter} (landmarks) VALUES (?)",
            (json.dumps(landmarksCollection),)
        )
        conn.commit()
    finally:
        conn.close()
        
#=============================================================================================================================================

#Function to interpolate snapshot if not enough frames were captured
def interpolateSnapshots(landmarkCollection):
    actualSnapshots = len(landmarkCollection)
    
    if actualSnapshots < Constants.COLLECTION_SNAPSHOTS:
        x = np.linspace(0, 1, actualSnapshots)
        xNew = np.linspace(0, 1, Constants.COLLECTION_SNAPSHOTS)
        
        interpolated = []
        
        for i in range(Constants.HAND_POINTS):
            coords = np.array([frame[i] for frame in landmarkCollection])
            fInterpret = interp1d(x, coords, axis=0, kind="linear", fill_value="extrapolate")
            interpolated.append(fInterpret(xNew))
            
        return np.array(interpolated).transpose(1, 0, 2).toList()
    
    return landmarkCollection

#=============================================================================================================================================

#Function to downsample if there are more frames than needed
def downsampleSnapshots(landmarksCollection):
    actualSnapshots = len(landmarksCollection)
    
    if actualSnapshots > Constants.COLLECTION_SNAPSHOTS:
        return [landmarksCollection[i] for i in np.linspace(0, actualSnapshots - 1, Constants.COLLECTION_SNAPSHOTS).astype(int)]
    
    return landmarksCollection

#=============================================================================================================================================

#Function to create AI model for letter detection
def createModel():
    model = Sequential([
        Flatten(input_shape=(Constants.COLLECTION_SNAPSHOTS, Constants.HAND_POINTS, Constants.COORD_POINTS)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(Constants.LETTERS), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

#=============================================================================================================================================

#Function to train the AI model
def trainModel():
    landmarkData = getLandmarks()

    x = []
    y = []
    
    minSamples = min(len(landmarkList) for landmarkList in landmarkData.values())
    
    print(f"Training on {minSamples} samples per letter to balance the dataset.")

    for letter, landmarkList in landmarkData.items():
        if len(landmarkList) >= minSamples:
            selected_samples = random.sample(landmarkList, minSamples)
        else:
            selected_samples = landmarkList

        for landmarksCollection in selected_samples:
            if len(landmarksCollection) == Constants.COLLECTION_SNAPSHOTS:
                landmarksArray = np.array(landmarksCollection).reshape(Constants.COLLECTION_SNAPSHOTS, Constants.HAND_POINTS, Constants.COORD_POINTS)
                
                x.append(landmarksArray)
                y.append(list(Constants.LETTERS.values()).index(letter))

    x = np.array(x)
    y = np.array(y)

    if x.size == 0:
        print("No data available! Please collect data first.")
        return
    
    x, y = shuffle(x, y, random_state=Constants.RANDOM_SEED)

    model = createModel()
    model.fit(x, y, epochs=2000, validation_split=0.2)
    model.save(Constants.MODEL_PATH)

    print("Model trained and saved!")

#=============================================================================================================================================