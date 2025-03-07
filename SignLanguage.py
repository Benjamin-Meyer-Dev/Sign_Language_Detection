import cv2
import mediapipe as mp
import pygetwindow as gw

#==============================================================================================================================

def getHandTypeAndOrientation(handLandmarks, handLabel):
    handSelection = handLabel.classification[0].label
    
    if handSelection == 'Left':
        color = (0, 0, 255)
    elif handSelection == 'Right':
        color = (0, 255, 0)

    thumbX = handLandmarks.landmark[4].x
    pinkyX = handLandmarks.landmark[20].x

    if (handSelection == 'Left' and thumbX > pinkyX) or (handSelection == 'Right' and thumbX < pinkyX):
        handOrientation = 'Front'
    else:
        handOrientation = 'Back'

    return f"{handSelection} Hand", handOrientation, color

#==============================================================================================================================

def signLanguage():
    cam = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cam.read()
        
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgbFrame)
        
        if result.multi_hand_landmarks:
            for handLandmarks, handLabel in zip(result.multi_hand_landmarks, result.multi_handedness):
                xMin, yMin = float('inf'), float('inf')
                xMax, yMax = float('-inf'), float('-inf')
                
                for lm in handLandmarks.landmark:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    xMin, yMin = min(xMin, x), min(yMin, y)
                    xMax, yMax = max(xMax, x), max(yMax, y)
                
                handType, handOrientation, color = getHandTypeAndOrientation(handLandmarks, handLabel)
                
                cv2.rectangle(frame, (xMin, yMin), (xMax, yMax), color, 2)
                cv2.putText(frame, f"{handType} - {handOrientation}", (xMin, yMin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Camera', frame)

        window = gw.getWindowsWithTitle('Camera')
        
        if window:
            window[0].activate()

        if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()

#==============================================================================================================================

if __name__ == "__main__":
    signLanguage()

#==============================================================================================================================