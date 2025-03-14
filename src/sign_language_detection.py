import cv2
import json
import os
import string
import sqlite3
import sys
import time

import mediapipe as mp
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

#=============================================================================================================================================

#Constants
COLLECTION_LENGTH = 2
COLLECTION_SNAPSHOTS = 34

DATABASE_PATH = "../database/sign_language.db"
MODEL_PATH = "../model/sign_language.h5"
LETTER_EXAMPLES_PATH = "Letter Examples"

LETTERS = {i: letter for i, letter in enumerate(string.ascii_uppercase[:5])}

#=============================================================================================================================================

#Function to access the database for collection count
def getCollectionCount(letter):
    conn = sqlite3.connect(DATABASE_PATH)
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
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    landmarkData = {}
    
    for letter in LETTERS.values():
        cursor.execute(f"SELECT landmarks FROM {letter}")
        rows = cursor.fetchall()
        landmarkData[letter] = [json.loads(row[0]) for row in rows]
        
    conn.close()
    
    return landmarkData

#=============================================================================================================================================

#Fucntion to insert landmark data into the database
def insertLandmarks(letter, landmarksCollection):
    conn = sqlite3.connect(DATABASE_PATH)
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
                    
                    if time.time() - startTime >= COLLECTION_LENGTH:
                        collecting = False
                        
                        insertLandmarks(letter, landmarksCollection)
                        
                        resetKey()
    
                        print(f"Captured {letter} collection {collectionNumber + 1}: Data stored in SQLite.")
        
        updateFrame(frame)

#=============================================================================================================================================

#Function to create AI model for letter detection
def createModel():
    model = Sequential([
        Flatten(input_shape=(COLLECTION_SNAPSHOTS, 21, 3)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(LETTERS), activation='softmax')
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
            if len(landmarksCollection) == COLLECTION_SNAPSHOTS:
                landmarksArray = np.array(landmarksCollection).reshape(COLLECTION_SNAPSHOTS, 21, 3)
                
                x.append(landmarksArray)
                y.append(list(LETTERS.values()).index(letter))

    x = np.array(x)
    y = np.array(y)

    if x.size == 0:
        print("No data available! Please collect data first.")
        return

    model = createModel()
    model.fit(x, y, epochs=2000, validation_split=0.2)
    model.save(MODEL_PATH)

    print("Model trained and saved!")

#=============================================================================================================================================

# def liveDetection():
#     model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    
#     if model is None:
#         print("No trained model found! Train the model first.")
#         return
        
#     while camera.isOpened():
#         ret, frame = camera.read()
        
#         if not ret:
#             break
        
#         frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frameRgb)

#         if results.multi_hand_landmarks:
#             for handLandmarks in results.multi_hand_landmarks:
#                 mpDrawing.draw_landmarks(frame, handLandmarks, mpHands.HAND_CONNECTIONS)
                
#                 landmarks = np.array([[lm.x, lm.y, lm.z] for lm in handLandmarks.landmark])
#                 landmarksArray = np.tile(landmarks, (COLLECTION_LENGTH, 1, 1))
                
#                 prediction = model.predict(landmarksArray.reshape(1, COLLECTION_LENGTH, 21, 3))
#                 action = LETTERS[np.argmax(prediction)]
                
#                 cv2.putText(frame, action, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
#         cv2.imshow("Sign Language Detection", frame)
        
#         if cv2.waitKey(1) & 0xFF == ord(' ') or cv2.getWindowProperty("Sign Language Detection", cv2.WND_PROP_VISIBLE) < 1:
#             break

#     camera.release()
#     cv2.destroyAllWindows()

#=============================================================================================================================================
