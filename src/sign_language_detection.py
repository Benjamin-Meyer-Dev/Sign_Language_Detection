import cv2
import constants
import datetime
import os
import sqlite3
import sys

import threading

import mediapipe as mp

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from video_output import liveDetection
from video_output import dataCollection

#=============================================================================================================================================

# GUI components class
class SignLanguageDetection(QMainWindow):
    
    #=============================================================================================================================================
    
    # Initialization class
    def __init__(self):
        super().__init__()
        
        with open(constants.GUI_STYLING_PATH, "r") as file:
            self.setStyleSheet(file.read())
        
        self.setWindowTitle("Sign Language Detection")        
        self._dragPos = None
        self.model = None
        
        self.initializeDatabase()
        self.generalSetup()
        
        self.setupMainWindow()
        self.setupTitleBar()
        
        self.setupContentWindow()
        
        self.setupVideoOutput()
        
        self.setupUserInteraction()
        self.setupExampleSignSection()
        self.setupAIOutputSection()
        
        self.setupFooter()
        
        self.addUIItems()
        
        self.setupCamera()
        
        if not os.path.exists(constants.MODEL_PATH):
            self.dataCollectionMode()
        else:
            self.liveDetectionMode()
    
    #=============================================================================================================================================
    
    # Function to initialize the database if needed
    def initializeDatabase(self):
        conn = sqlite3.connect(constants.DATABASE_PATH)
        cursor = conn.cursor()

        for letter in constants.LETTERS.values():
            cursor.execute(
                f'''
                CREATE TABLE IF NOT EXISTS {letter} (
                    collectionSet INTEGER PRIMARY KEY AUTOINCREMENT,
                    landmarks TEXT
                )
                '''
            )
            
            cursor.execute(f"DELETE FROM {letter} WHERE collectionSet > 30")

        conn.commit()
        conn.close()
        
    #=============================================================================================================================================
    
    # Function to setup items for general program use
    def generalSetup(self):
        self.camera = cv2.VideoCapture(0)
        
        self.frame = None
        self.currentKey = None
        
        self.mpHands = mp.solutions.hands
        self.mpDrawing = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    #=============================================================================================================================================

    # Function to set up the main window layout
    def setupMainWindow(self):
        self.setGeometry(100, 100, 800, 480)
        self.setWindowFlag(Qt.FramelessWindowHint)

        self.masterWidget = QWidget()
        self.setCentralWidget(self.masterWidget)

        self.masterLayout = QVBoxLayout(self.masterWidget)
        self.masterLayout.setContentsMargins(0, 0, 0, 0)

        self.mainWidget = QWidget(self)
        self.mainWidget.setObjectName("mainWidget")

        self.mainLayout = QVBoxLayout(self.mainWidget)
        self.mainLayout.setContentsMargins(10, 0, 10, 0)
        self.mainLayout.setSpacing(0)

    #=============================================================================================================================================
    
    # Function to set up the title bar
    def setupTitleBar(self):
        self.titleBar = QWidget(self)
        self.titleBar.setFixedHeight(constants.TITLE_BAR_HEIGHT)

        self.titleLayout = QHBoxLayout(self.titleBar)
        self.titleLayout.setContentsMargins(0, 0, 0, 0)
        self.titleLayout.setAlignment(Qt.AlignVCenter)
        
        self.programIcon = QLabel(self.titleBar)
        self.programIcon.setObjectName("programIcon")
        self.programIcon.setFixedSize(constants.PROGRAM_ICON_SIZE, constants.PROGRAM_ICON_SIZE)
        self.programIcon.setPixmap(QPixmap(constants.PROGRAM_ICON_PATH))
        self.programIcon.setScaledContents(True)

        self.titleLabel = QLabel("Sign Language Detection")
        self.titleLabel.setObjectName("titleLabel")

        self.exitButton = QPushButton(self)
        self.exitButton.setObjectName("exitButton")
        self.exitButton.setFixedSize(constants.EXIT_BUTTON_SIZE, constants.EXIT_BUTTON_SIZE)
        self.exitButton.setIcon(QIcon(constants.EXIT_ICON_PATH))
        self.exitButton.setIconSize(QSize(constants.EXIT_BUTTON_SIZE - 2, constants.EXIT_BUTTON_SIZE - 2))
        self.exitButton.setFlat(True)
        self.exitButton.clicked.connect(self.close)

    #=============================================================================================================================================
    
    # Function to set up the content window layout
    def setupContentWindow(self):
        self.contentSection = QWidget(self)

        self.contentLayout = QHBoxLayout(self.contentSection)
        self.contentLayout.setContentsMargins(0, 0, 0, 0)
        
    #=============================================================================================================================================
    
    # Function to set up the video output
    def setupVideoOutput(self):
        self.videoOutput = QWidget(self.contentSection)
        
        self.videoLayout = QVBoxLayout(self.videoOutput)
        self.videoLayout.setContentsMargins(0, 0, 0, 0)

        self.videoLabel = QLabel(self.videoOutput)

    #=============================================================================================================================================
    
    # Function to set up the user interaction section
    def setupUserInteraction(self):
        self.userItems = QWidget(self.contentSection)

        self.userLayout = QVBoxLayout(self.userItems)
        self.userLayout.setAlignment(Qt.AlignVCenter)

    #=============================================================================================================================================
    
    # Function to set up the example sign section
    def setupExampleSignSection(self):
        self.exampleLabel = QLabel("Sign Language Letters:", self.userItems)
        self.exampleLabel.setObjectName("exampleLabel")

        self.exampleWindow = QWidget(self.userItems)
        self.exampleWindow.setObjectName("exampleWindow")

        self.exampleLayout = QVBoxLayout(self.exampleWindow)
        self.exampleLayout.setAlignment(Qt.AlignVCenter)

        self.exampleImage = QLabel(self.exampleWindow)
        self.exampleImage.setPixmap(QPixmap())
        self.exampleImage.setFixedSize(135, 140)

        self.dropdownMenu = QComboBox(self.exampleWindow)
        self.dropdownMenu.addItems(constants.LETTERS.values())
        self.dropdownMenu.setFocusPolicy(Qt.NoFocus)
        self.dropdownMenu.currentIndexChanged.connect(lambda: self.updateExampleAIImage(self.dropdownMenu.currentText(), self.exampleImage))

        self.updateExampleAIImage(self.dropdownMenu.currentText(), self.exampleImage)

    #=============================================================================================================================================
    
    # Function to set up the AI output section
    def setupAIOutputSection(self):
        self.AILabel = QLabel("AI Letter Guess:", self.userItems)
        self.AILabel.setObjectName("AILabel")
        
        self.AIWindow = QWidget(self.userItems)
        self.AIWindow.setObjectName("AIWindow")
        
        self.AILayout = QVBoxLayout(self.AIWindow)
        self.AILayout.setAlignment(Qt.AlignVCenter)
        
        self.AIImage = QLabel(self.AIWindow)
        self.AIImage.setPixmap(QPixmap())
        self.AIImage.setFixedSize(135, 140)

    #=============================================================================================================================================
    
    # Function to set up the footer section
    def setupFooter(self):
        self.footer = QWidget(self)
        self.footer.setFixedHeight(constants.FOOTER_HEIGHT)

        self.footerLayout = QHBoxLayout(self.footer)
        self.footerLayout.setContentsMargins(0, 0, 0, 0)
        self.footerLayout.setAlignment(Qt.AlignVCenter)

        self.infoText = QLabel()
        self.infoText.setObjectName("infoText")
        self.infoText.setAlignment(Qt.AlignLeft)

        self.author = QLabel(f"\u00A9 {datetime.datetime.now().year} Benjamin Meyer")
        self.author.setObjectName("author")
        self.author.setAlignment(Qt.AlignRight)
    
    #=============================================================================================================================================
    
    # Function add all of the GUI items
    def addUIItems(self):
        self.titleLayout.addWidget(self.programIcon)
        self.titleLayout.addWidget(self.titleLabel)
        self.titleLayout.addStretch()
        self.titleLayout.addWidget(self.exitButton)
        
        self.videoLayout.addWidget(self.videoLabel)
        
        self.exampleLayout.addWidget(self.dropdownMenu)
        self.exampleLayout.addWidget(self.exampleImage)
        
        self.AILayout.addWidget(self.AIImage)
        
        self.userLayout.addWidget(self.exampleLabel)
        self.userLayout.addWidget(self.exampleWindow)
        self.userLayout.addWidget(self.AILabel)
        self.userLayout.addWidget(self.AIWindow)
        
        self.contentLayout.addWidget(self.videoOutput)
        self.contentLayout.addWidget(self.userItems)
        
        self.footerLayout.addWidget(self.infoText)
        self.footerLayout.addWidget(self.author)
        
        self.mainLayout.addWidget(self.titleBar)
        self.mainLayout.addWidget(self.contentSection)
        self.mainLayout.addWidget(self.footer)
        
        self.masterLayout.addWidget(self.mainWidget)
        
    #=============================================================================================================================================

    # Event handler for mouse press
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragPos = event.globalPos()
            event.accept()

    #=============================================================================================================================================

    # Event handler for mouse move event
    def mouseMoveEvent(self, event):
        if self._dragPos is not None:
            delta = event.globalPos() - self._dragPos
            self.move(self.pos() + delta)
            self._dragPos = event.globalPos()
            event.accept()

    #=============================================================================================================================================

    # Handler for mouse release event
    def mouseReleaseEvent(self, event):
        self._dragPos = None
        event.accept()
    
    #=============================================================================================================================================

    # Handler for key press event
    def keyPressEvent(self, event):
        key = event.key()
        
        if key == constants.ENTER_KEYCODE:
            self.currentKey = 'Enter'
        elif chr(key).upper() in constants.LETTERS.values():
            self.currentKey = chr(key).upper()
        else:
            self.currentKey = None

    #=============================================================================================================================================
    
    # Function to set up the camera and timer
    def setupCamera(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.cameraOutput)
        self.timer.start(constants.WEBCAM_UPDATE)

    #=============================================================================================================================================
    
    # Function used to update the video output
    def cameraOutput(self):
        if self.frame is None:
            return
        
        frameRgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image = QImage(frameRgb.data, frameRgb.shape[1], frameRgb.shape[0], frameRgb.strides[0], QImage.Format_RGB888)
        
        self.videoLabel.setPixmap(QPixmap.fromImage(image))
        
    #=============================================================================================================================================

    # Function to update the video output frame
    def updateFrame(self, frame):
        self.frame = frame
        
    #=============================================================================================================================================
    
    #Function to run the live detection mode
    def liveDetectionMode(self):
        self.activeThread = threading.Thread(target=liveDetection, args=(self.camera, self.hands, self.AIImage, self.updateExampleAIImage, self.updateFrame), daemon=True)
        self.activeThread.start()
        
        self.infoText.setText("AI now detecting sign language letters.")
        
    #=============================================================================================================================================
    
    #Function to update the example and AI guess images
    def updateExampleAIImage(self, letter, imageLabel):
        if not letter or not os.path.exists(os.path.join(constants.LETTER_EXAMPELS_PATH, f"{letter.upper()}.png")):
            imageLabel.clear()
            return
        
        imagePath = os.path.join(constants.LETTER_EXAMPELS_PATH, f"{letter.upper()}.png")
        pixmap = QPixmap(imagePath)
        imageLabel.setPixmap(pixmap.scaled(imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    #=============================================================================================================================================

    # Function to run the data collection mode
    def dataCollectionMode(self):
        self.activeThread = threading.Thread(target=dataCollection, args=(self.camera, self.hands, self.mpDrawing, self.mpHands, self.getKeyPress, self.setCurrentKey, self.updateFrame,), daemon=True)
        self.activeThread.start()
        
        self.infoText.setText("Now collecting new letter data.")
        
    #=============================================================================================================================================

    # Function to get the last pressed key
    def getKeyPress(self):
        return self.currentKey
    
    #=============================================================================================================================================

    # Function to set the last pressed key to none
    def setCurrentKey(self):
        self.currentKey = None

#=============================================================================================================================================

if __name__ == "__main__":    
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(constants.PROGRAM_ICON_PATH))
    
    window = SignLanguageDetection()
    window.show()
    
    sys.exit(app.exec_())
    
#=============================================================================================================================================