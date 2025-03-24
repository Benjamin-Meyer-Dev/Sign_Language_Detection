import string

#=============================================================================================================================================

# Constants used throughout the program
COORD_POINTS = 3
HAND_POINTS = 21
RANDOM_SEED = 42

COLLECTION_LENGTH = 2
COLLECTION_SNAPSHOTS = 30
CONFIDENCE_CHECKS = 5
CONFIDENCE_THRESHOLD = 0.95
DROPOUT = 0.2
EPOCHS = 1000
REGULARIZER = 0.001

EXIT_BUTTON_SIZE = 20
FOOTER_HEIGHT = 30
PROGRAM_ICON_SIZE = 25
TITLE_BAR_HEIGHT = 40

A_KEYCODE = 65
Z_KEYCODE = 90
ENTER_KEYCODE = 16777220

WEBCAM_UPDATE = 60

DATABASE_PATH = "../database/sign_language.db"
EXIT_ICON_PATH = "../images/exit_icon.svg"
GUI_STYLING_PATH = "./qss/gui.qss"
LETTER_EXAMPELS_PATH = "../images/Letter Examples"
MODEL_PATH = "../model/sign_language_model.keras"
PROGRAM_ICON_PATH = "../images/program_icon.svg"

LETTERS = {i: letter for i, letter in enumerate(string.ascii_uppercase)}

#=============================================================================================================================================