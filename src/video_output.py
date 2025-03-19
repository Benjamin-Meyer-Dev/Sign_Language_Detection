import cv2
import constants
import json
import random
import sqlite3
import time

import numpy as np
import tensorflow as tf

from scipy.interpolate import interp1d
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential

#=============================================================================================================================================

# Function for live sign language detection
def liveDetection(camera, hands, imageLocation, updateImage, updateFrame):    
    model = tf.keras.models.load_model(constants.MODEL_PATH)
    snapshots = []
    lastLetter = None
    
    while camera.isOpened():
        ret, frame = camera.read()
        
        if not ret:
            break
        
        frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRgb)

        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in handLandmarks.landmark])
                snapshots.append(landmarks)
                
                if len(snapshots) == constants.COLLECTION_SNAPSHOTS:
                    landmarksArray = np.array(snapshots).reshape(constants.COLLECTION_SNAPSHOTS, constants.HAND_POINTS, constants.COORD_POINTS)

                    prediction = model.predict(landmarksArray.reshape(1, constants.COLLECTION_SNAPSHOTS, constants.HAND_POINTS, constants.COORD_POINTS))[0]  
                    
                    topPrediction = np.argmax(prediction)
                    bestLetter = constants.LETTERS[topPrediction]
                    bestConfidence = prediction[topPrediction]
                    
                    if bestConfidence < constants.CONFIDENCE_THRESHOLD:
                        bestLetter = None

                    if bestLetter != lastLetter:
                        lastLetter = bestLetter
                    
                    updateImage(lastLetter, imageLocation)
                    snapshots = []
        else:
            updateImage(None, imageLocation)
        
        updateFrame(frame)

#=============================================================================================================================================

# Function to collect data for training
def dataCollection(camera, hands, mpDrawing, mpHands, getKeyPress, resetKey, updateFrame):    
    landmarksCollection = []
    collecting = False
    letter = None
    collectionNumber = 0
    
    while camera.isOpened():
        ret, frame = camera.read()
        
        if not ret:
            break
        
        frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRgb)

        if collecting:
            cv2.putText(frame, "Collecting data...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            cv2.putText(frame, "Press 'Enter' to train AI model.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            if results.multi_hand_landmarks:
                cv2.putText(frame, "Press a letter to collect data.", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if not collecting:
            letter = getKeyPress()
            
            if letter == 'Enter':
                createModel()
                resetKey()
                continue

        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mpDrawing.draw_landmarks(frame, handLandmarks, mpHands.HAND_CONNECTIONS)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in handLandmarks.landmark])

                if not collecting:
                    letter = getKeyPress()

                if letter and letter != 'Enter' and not collecting:
                    print(f"Start collecting data for: {letter}")

                    landmarksCollection = []
                    collecting = True

                    startTime = time.time()
                    endTime = startTime + constants.COLLECTION_LENGTH

                    collectionNumber = getCollectionCount(letter) + 1

                    frameCount = 0
                    interval = constants.COLLECTION_LENGTH / constants.COLLECTION_SNAPSHOTS

                    nextCaptureTime = startTime

                if collecting:
                    currentTime = time.time()

                    if currentTime >= nextCaptureTime:
                        landmarksCollection.append(landmarks.tolist())
                        frameCount += 1
                        nextCaptureTime += interval

                    if currentTime >= endTime or frameCount >= constants.COLLECTION_SNAPSHOTS:
                        collecting = False

                        if len(landmarksCollection) < constants.COLLECTION_SNAPSHOTS:
                            landmarksCollection = interpolateSnapshots(landmarksCollection)
                        elif len(landmarksCollection) > constants.COLLECTION_SNAPSHOTS:
                            landmarksCollection = downsampleSnapshots(landmarksCollection)

                        insertLandmarks(letter, landmarksCollection)
                        resetKey()

                        print(f"Captured {letter} collection {collectionNumber}: {len(landmarksCollection)} snapshots stored in SQLite.")

        updateFrame(frame)

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

#Function to interpolate snapshot if not enough frames were captured
def interpolateSnapshots(landmarkCollection):
    actualSnapshots = len(landmarkCollection)
    
    if actualSnapshots < constants.COLLECTION_SNAPSHOTS:
        x = np.linspace(0, 1, actualSnapshots)
        xNew = np.linspace(0, 1, constants.COLLECTION_SNAPSHOTS)
        
        interpolated = []
        
        for i in range(constants.HAND_POINTS):
            coords = np.array([frame[i] for frame in landmarkCollection])
            fInterpret = interp1d(x, coords, axis=0, kind="linear", fill_value="extrapolate")
            interpolated.append(fInterpret(xNew))
            
        return np.array(interpolated).transpose(1, 0, 2).tolist()
    
    return landmarkCollection

#=============================================================================================================================================

#Function to downsample if there are more frames than needed
def downsampleSnapshots(landmarksCollection):
    actualSnapshots = len(landmarksCollection)
    
    if actualSnapshots > constants.COLLECTION_SNAPSHOTS:
        return [landmarksCollection[i] for i in np.linspace(0, actualSnapshots - 1, constants.COLLECTION_SNAPSHOTS).astype(int)]
    
    return landmarksCollection

#=============================================================================================================================================

# Function to create the AI model and automatically train it
def createModel():
    print("Creating model...")
    
    model = Sequential([
        Input(shape=(constants.COLLECTION_SNAPSHOTS, constants.HAND_POINTS, constants.COORD_POINTS)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(len(constants.LETTERS), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Model created successfully!")
    trainModel(model)

    return model

#=============================================================================================================================================

# Function to train the AI model
def trainModel(model):
    print("Training model...")

    landmarkData = getLandmarks()
    x, y = [], []
    
    minSamples = min(len(landmarkList) for landmarkList in landmarkData.values())
    
    if minSamples == 0:
        print("Error: Insufficient data! At least one sample is required per letter.")
        return
    
    print(f"Training on {minSamples} samples per letter to balance the dataset.")

    for letter, landmarkList in landmarkData.items():
        if len(landmarkList) >= minSamples:
            selected_samples = random.sample(landmarkList, minSamples)
        else:
            selected_samples = landmarkList

        for landmarksCollection in selected_samples:
            if len(landmarksCollection) == constants.COLLECTION_SNAPSHOTS:
                landmarksArray = np.array(landmarksCollection).reshape(constants.COLLECTION_SNAPSHOTS, constants.HAND_POINTS, constants.COORD_POINTS)
                
                x.append(landmarksArray)
                y.append(list(constants.LETTERS.values()).index(letter))

    x, y = np.array(x), np.array(y)

    if x.size == 0:
        print("No data available! Please collect data first.")
        return
    
    x, y = shuffle(x, y, random_state=constants.RANDOM_SEED)

    model.fit(x, y, epochs=constants.EPOCHS, validation_split=0.2)
    model.save(constants.MODEL_PATH)

    print("Model trained and saved successfully!")

#=============================================================================================================================================