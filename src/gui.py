import cv2
import datetime
import os

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

#=============================================================================================================================================

#Constants
WEBCAM_UPDATE = 60
TITLE_BAR_HEIGHT = 40
EXIT_BUTTON_SIZE = 20
TOGGLE_BUTTONS_HEIGHT = 45
FOOTER_HEIGHT = 30

EXIT_ICON_PATH = "../Images/exit_icon.svg"
LETTER_EXAMPELS_PATH = "../Images/Letter Examples"
STYLING_PATH = "./QSS/gui.qss"

#=============================================================================================================================================

#GUI components class
class GUI(QMainWindow):
    
    #=============================================================================================================================================
    
    #Initialization class
    def __init__(self):
        super().__init__()
        
        self._drag_pos = None

        with open(STYLING_PATH, "r") as file:
            self.setStyleSheet(file.read())

        #=========================================================================
        
        #Main window setup
        self.setGeometry(100, 100, 800, 480)
        self.setWindowFlag(Qt.FramelessWindowHint)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.content_widgit = QWidget(self)
        self.content_widgit.setObjectName("content_widgit")
        
        self.content_layout = QVBoxLayout(self.content_widgit)
        self.content_layout.setContentsMargins(10, 0, 10, 0)
        self.content_layout.setSpacing(0)
        
        self.main_layout.addWidget(self.content_widgit)
        
        #=========================================================================
        
        #Custom title bar
        self.title_bar = QWidget(self)
        self.title_bar.setFixedHeight(TITLE_BAR_HEIGHT)
        
        self.title_layout = QHBoxLayout(self.title_bar)
        self.title_layout.setContentsMargins(0, 0, 0, 0)
        self.title_layout.setAlignment(Qt.AlignVCenter)
        
        self.title_label = QLabel("Sign Language Detection")
        self.title_label.setObjectName("title_label")
        
        self.exit_button = QPushButton(self)
        self.exit_button.setObjectName("exit_button")
        self.exit_button.setFixedSize(EXIT_BUTTON_SIZE, EXIT_BUTTON_SIZE)
        self.exit_button.setIcon(QIcon(EXIT_ICON_PATH))
        self.exit_button.setIconSize(QSize(EXIT_BUTTON_SIZE - 2, EXIT_BUTTON_SIZE - 2))
        self.exit_button.setFlat(True)
        self.exit_button.clicked.connect(self.close)
        
        #=========================================================================
        
        #Content window
        self.output = QWidget(self)
        
        self.output_layout = QHBoxLayout(self.output)
        self.output_layout.setContentsMargins(0, 0, 0, 0)
        
        #=========================================================================
        
        #Video output
        self.video_output = QWidget(self.output)
        
        self.video_layout = QVBoxLayout(self.video_output)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel(self.video_output)
        
        #=========================================================================
        
        #Buttons section
        self.buttons = QWidget(self)
        self.buttons.setFixedHeight(TOGGLE_BUTTONS_HEIGHT)
        self.buttons.setObjectName("buttons")
        
        self.buttons_layout = QHBoxLayout(self.buttons)
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.buttons_layout.setAlignment(Qt.AlignVCenter)
        self.buttons_layout.setStretch(0, 1)
        self.buttons_layout.setStretch(1, 1)
        self.buttons_layout.setStretch(2, 1)
        
        self.live_detection_button = QPushButton(self.buttons)
        self.live_detection_button.setObjectName("live_detection_button")
        self.live_detection_button.setText("Live Detection")
        self.live_detection_button.setFixedHeight(TOGGLE_BUTTONS_HEIGHT)
        self.live_detection_button.clicked.connect(lambda: self.update_info_text("Live Detection"))
        
        self.collect_data_button = QPushButton(self.buttons)
        self.collect_data_button.setObjectName("collect_data_button")
        self.collect_data_button.setText("Collect Data")
        self.collect_data_button.setFixedHeight(TOGGLE_BUTTONS_HEIGHT)
        self.collect_data_button.clicked.connect(lambda: self.update_info_text("Data Collection"))
        
        self.no_detection_button = QPushButton(self.buttons)
        self.no_detection_button.setObjectName("no_detection_button")
        self.no_detection_button.setText("No Detection")
        self.no_detection_button.setFixedHeight(TOGGLE_BUTTONS_HEIGHT)
        self.no_detection_button.clicked.connect(lambda: self.update_info_text("No Detection"))
        
        #=========================================================================
        
        #User interaction window
        self.user_items = QWidget(self.output)
        
        self.user_layout = QVBoxLayout(self.user_items)
        self.user_layout.setAlignment(Qt.AlignVCenter)
        
        #=========================================================================
        
        #Example sign section
        self.example_label = QLabel("Example Signs:", self.user_items)
        self.example_label.setObjectName("example_label")
        
        self.example_window = QWidget(self.user_items)
        self.example_window.setObjectName("example_window")
        
        self.example_layout = QVBoxLayout(self.example_window)
        self.example_layout.setAlignment(Qt.AlignVCenter)
        
        self.example_image = QLabel(self.example_window)
        self.example_image.setPixmap(QPixmap())
        self.example_image.setFixedSize(135, 140)
        
        self.dropdown_menu = QComboBox(self.example_window)
        self.dropdown_menu.addItems([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        self.dropdown_menu.currentIndexChanged.connect(lambda: self.update_image(self.dropdown_menu.currentText(), self.example_image))
        
        self.update_image(self.dropdown_menu.currentText(), self.example_image)
        
        #=========================================================================
        
        #AI output section
        self.ai_guess_label = QLabel("AI Guess:", self.user_items)
        self.ai_guess_label.setObjectName("ai_label")
        
        self.ai_output = QWidget(self.user_items)
        self.ai_output.setObjectName("ai_output")
        
        self.ai_layout = QVBoxLayout(self.ai_output)
        self.ai_layout.setAlignment(Qt.AlignVCenter)
        
        self.ai_guess_image = QLabel(self.user_items)
        self.ai_guess_image.setPixmap(QPixmap())
        self.ai_guess_image.setFixedSize(135, 140)
        
        self.update_image('A', self.ai_guess_image)
        
        #=========================================================================
        
        #Footer section
        self.footer = QWidget(self)
        self.footer.setFixedHeight(FOOTER_HEIGHT)
        
        self.footer_layout = QHBoxLayout(self.footer)
        self.footer_layout.setContentsMargins(0, 0, 0, 0)
        self.footer_layout.setAlignment(Qt.AlignVCenter)
        
        self.info_text = QLabel()
        self.info_text.setObjectName("info_text")
        self.info_text.setAlignment(Qt.AlignLeft)
        
        self.author = QLabel(f"\u00A9 {datetime.datetime.now().year} Benjamin Meyer")
        self.author.setObjectName("author")
        self.author.setAlignment(Qt.AlignRight)
        
        #=========================================================================
        
        #Add sections to the layouts
        self.title_layout.addWidget(self.title_label)
        self.title_layout.addWidget(self.exit_button)
        
        self.buttons_layout.addWidget(self.live_detection_button)
        self.buttons_layout.addWidget(self.collect_data_button)
        self.buttons_layout.addWidget(self.no_detection_button)
        
        self.video_layout.addWidget(self.video_label)
        self.video_layout.addWidget(self.buttons)
        
        self.example_layout.addWidget(self.dropdown_menu)
        self.example_layout.addWidget(self.example_image)
        
        self.ai_layout.addWidget(self.ai_guess_image)
        
        self.user_layout.addWidget(self.example_label)
        self.user_layout.addWidget(self.example_window)
        self.user_layout.addWidget(self.ai_guess_label)
        self.user_layout.addWidget(self.ai_output)
        
        self.output_layout.addWidget(self.video_output)
        self.output_layout.addWidget(self.user_items)
        
        self.footer_layout.addWidget(self.info_text)
        self.footer_layout.addWidget(self.author)
        
        self.content_layout.addWidget(self.title_bar)
        self.content_layout.addWidget(self.output)
        self.content_layout.addWidget(self.footer)
        
        #=========================================================================
        
        #Camera setup
        self.camera = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(WEBCAM_UPDATE)
    
    #=============================================================================================================================================
    
    #Function to update the example and ai guess images
    def update_image(self, letter, image_label):
        image_path = os.path.join(LETTER_EXAMPELS_PATH, f"{letter.upper()}.png")
        
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            image_label.setPixmap(pixmap.scaled(image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            image_label.clear()
            
    #=============================================================================================================================================
    
    # Function to update info text
    def update_info_text(self, message):
        self.info_text.setText(f"Mode: {message}")

    #=============================================================================================================================================
    
    #Function used to update the video output
    def update_frame(self):        
        ret, frame = self.camera.read()
        
        if not ret:
            return
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        
        self.video_label.setPixmap(QPixmap.fromImage(image))

    #=============================================================================================================================================

    #Event handler for mouse press
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos()
            event.accept()

    #=============================================================================================================================================

    #Event handler for mouse move event
    def mouseMoveEvent(self, event):
        if self._drag_pos is not None:
            delta = event.globalPos() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPos()
            event.accept()

    #=============================================================================================================================================

    #Handler for mouse release event
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()

#=============================================================================================================================================
