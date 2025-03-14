import cv2
import constants
import json
import os

import sqlite3
import time

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

#=============================================================================================================================================

#Function to access the database for collection count
def getCollectionCount(letter):
    conn = sqlite3.connect(constants.DATABASE_PATH)
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
    conn = sqlite3.connect(constants.DATABASE_PATH)
    cursor = conn.cursor()
    
    landmarkData = {}
    
    for letter in constants.LETTERS.values():
        cursor.execute(f"SELECT landmarks FROM {letter}")
        rows = cursor.fetchall()
        landmarkData[letter] = [json.loads(row[0]) for row in rows]
        
    conn.close()
    
    return landmarkData

#=============================================================================================================================================

#Fucntion to insert landmark data into the database
def insertLandmarks(letter, landmarksCollection):
    conn = sqlite3.connect(constants.DATABASE_PATH)
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

#Function for normal video display
def baseMode(camera, updateFrame, stopEvent):
    while camera.isOpened():
        if stopEvent.is_set():
            break
        
        ret, frame = camera.read()
        
        if not ret:
            break
        
        updateFrame(frame)

#=============================================================================================================================================

#Function to collect data for training
def collectionMode(camera, hands, mpDrawing, mpHands, updateFrame, getKeyPress, resetKey, stopEvent):    
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
                    startTime = time.time()
                    collecting = True
                    
                    collectionCount = getCollectionCount(letter)
                    collectionNumber = collectionCount + 1
                    
                if collecting:
                    landmarksCollection.append(landmarks.tolist())
                    
                    if time.time() - startTime >= constants.COLLECTION_LENGTH:
                        collecting = False
                        
                        insertLandmarks(letter, landmarksCollection)
                        
                        resetKey()
    
                        print(f"Captured {letter} collection {collectionNumber + 1}: Data stored in SQLite.")
        
        updateFrame(frame)
        
#=============================================================================================================================================

#Function for live sign language detection
def liveDetection(camera, hands, mpDrawing, mpHands, updateFrame, stopEvent):
    model = tf.keras.models.load_model(constants.MODEL_PATH) if os.path.exists(constants.MODEL_PATH) else None
    
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
                landmarksArray = np.tile(landmarks, (constants.COLLECTION_LENGTH, 1, 1))
                
                prediction = model.predict(landmarksArray.reshape(1, constants.COLLECTION_LENGTH, 21, 3))
                action = constants.LETTERS[np.argmax(prediction)]
                
                cv2.putText(frame, action, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
        updateFrame(frame)

#=============================================================================================================================================

#Function to create AI model for letter detection
def createModel():
    model = Sequential([
        Flatten(input_shape=(constants.COLLECTION_SNAPSHOTS, 21, 3)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(constants.LETTERS), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

#=============================================================================================================================================

#Function to train the AI model
def trainModel():
    landmarkData = getLandmarks()

    x = []
    y = []

    for letter, landmarkList in landmarkData.items():
        for landmarksCollection in landmarkList:
            if len(landmarksCollection) == constants.COLLECTION_SNAPSHOTS:
                landmarksArray = np.array(landmarksCollection).reshape(constants.COLLECTION_SNAPSHOTS, 21, 3)
                
                x.append(landmarksArray)
                y.append(list(constants.LETTERS.values()).index(letter))

    x = np.array(x)
    y = np.array(y)

    if x.size == 0:
        print("No data available! Please collect data first.")
        return

    model = createModel()
    model.fit(x, y, epochs=2000, validation_split=0.2)
    model.save(constants.MODEL_PATH)

    print("Model trained and saved!")

#=============================================================================================================================================