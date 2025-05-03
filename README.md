# Sign Language Letter Detection

## Table of contents
- [General Info](#general-info)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

## General Info
This is a real-time sign language letter detection tool that uses a trained TensorFlow model to interpret hand gestures into alphabet letters. It utilizes hand landmark data captured via MediaPipe and allows users to either use a pre-trained model or collect and train their own using a built-in interface.

---

## Features

- Predicts sign language alphabet letters using hand position data.
- Custom UI with a camera feed, letter examples and an AI guess section.
- SQLite database that stores the pre-trained model's training data.
- Option to delete the pre-trained model and create a custom model by collecting your own training data.
- Works with cameras of various frame rates and supports varied amounts of training data per letter.

---

## Technologies Used
This project is created using the following technologies:

- **mediapipe** `0.10.9`
- **numpy** `1.26.4`
- **opencv-python** `4.11.0.86`
- **PyQt5** `5.15.11`
- **scikit-learn** `1.6.1`
- **scipy** `1.15.2`
- **tensorflow** `2.18.0`

---

## How to Run
Follow the steps below to run the project:

1. Clone the Repository
2. Create a Virtual Environment
3. Install Dependencies
4. Run the Application

Note:
- To gather your own training data and create your own model, simply delete the current model out of the "/model" folder before running the application.
