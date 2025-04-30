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
- SQLite database that houses the pre-trained model training data.
- User has the option to create a custom model by deleting the pre-trained one and collecting their own data using a varied UI.
- Flexibility around cameras of varying frame rates, and different amount of training data per letter.

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
