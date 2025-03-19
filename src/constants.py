import string

#=============================================================================================================================================

#Constants used throughout the program
COORD_POINTS = 3
HAND_POINTS = 21
RANDOM_SEED = 42

COLLECTION_LENGTH = 2
COLLECTION_SNAPSHOTS = 30
EPOCHS = 1000

EXIT_BUTTON_SIZE = 20
FOOTER_HEIGHT = 30
PROGRAM_ICON_SIZE = 25
TITLE_BAR_HEIGHT = 40

ENTER_KEYCODE = 16777220
WEBCAM_UPDATE = 60

DATABASE_PATH = "../Database/sign_language.db"
EXIT_ICON_PATH = "../Images/exit_icon.svg"
GUI_STYLING_PATH = "./QSS/gui.qss"
LETTER_EXAMPELS_PATH = "../Images/Letter Examples"
MODEL_PATH = "../Model/sign_language_model.keras"
PROGRAM_ICON_PATH = "../Images/program_icon.svg"

LETTERS = {i: letter for i, letter in enumerate(string.ascii_uppercase[:9])}

#=============================================================================================================================================