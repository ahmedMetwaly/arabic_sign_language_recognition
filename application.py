import base64
import pickle
import cv2
import os

import mediapipe as mp  # Import mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score  # Accuracy metrics
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS
from flask import Flask, request, jsonify, render_template, send_from_directory
import simplejpeg
import numpy as np
import eventlet
eventlet.monkey_patch()

from utils import decode_image_base64


app = Flask(__name__)
app.config['SECRET_KEY'] = 'somethingsoamazingomg!!!'
app.config["DEBUG"] = False
app.config["environment"] = "production"
socketio = SocketIO(app, message_queue='redis://redis:6379')


with open('signs4.pkl', 'rb') as f:
    model = pickle.load(f)


def isInitialized(hand):
    try:
        if hand.IsInitialized() == True:
            return True
    except:
        return False


@app.route('/')
def home():
    return render_template('camera.html')


@socketio.on('upload')
def predict(image_data):
    img_binary_str = decode_image_base64(image_data)
    image = simplejpeg.decode_jpeg(img_binary_str)

    # Make Detections
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        results = holistic.process(image)
        # print(results.face_landmarks)

    # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

    # 1. Draw face landmarks
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                            mp_drawing.DrawingSpec(
                                color=(80, 110, 10), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(
                                color=(80, 256, 121), thickness=1, circle_radius=1)
                            )

    # 2. Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(
                                color=(80, 22, 10), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(
                                color=(80, 44, 121), thickness=2, circle_radius=2)
                            )

    # 3. Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(
                                color=(121, 22, 76), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(
                                color=(121, 44, 250), thickness=2, circle_radius=2)
                            )

    # 4. Pose Detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(
                                color=(245, 117, 66), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(
                                color=(245, 66, 230), thickness=2, circle_radius=2)
                            )

    # Extract Pose landmarks
    if not isInitialized(results.pose_landmarks):
        emit("speak", 'move back to appear')

    # Extract left_hand landmarks
    elif not isInitialized(results.left_hand_landmarks):
        emit("speak", "move back one step to detect left hand")

    # Exract right_hand landmarks
    elif not isInitialized(results.right_hand_landmarks):
        emit("speak", "move back one step to detect right hand")

    # Concate rows
    elif isInitialized(results.pose_landmarks) and isInitialized(results.right_hand_landmarks) and isInitialized(results.left_hand_landmarks):
        # pose detection
        rowRLH = []

        pose = results.pose_landmarks.landmark
        for landmark in pose:
            rowRLH.append(landmark.x)
            rowRLH.append(landmark.y)
            rowRLH.append(landmark.z)
            rowRLH.append(landmark.visibility)
        # left hand coordinates
        left_hand = results.left_hand_landmarks.landmark
        for landmark in left_hand:
            rowRLH.append(landmark.x)
            rowRLH.append(landmark.y)
            rowRLH.append(landmark.z)
            rowRLH.append(landmark.visibility)
        # right hand coordinates
        right_hand = results.right_hand_landmarks.landmark
        for landmark in right_hand:
            rowRLH.append(landmark.x)
            rowRLH.append(landmark.y)
            rowRLH.append(landmark.z)
            rowRLH.append(landmark.visibility)

        # Make Detections
        X = pd.DataFrame([rowRLH])
        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]

        emit("speak", body_language_class)

        # Grab ear coords
        coords = tuple(np.multiply(
            np.array(
                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)), [640, 480]).astype(int))

        cv2.rectangle(image,
                    (coords[0], coords[1]+5),
                    (coords[0]+len(body_language_class)
                    * 20, coords[1]-30),
                    (245, 117, 16), -1)
        cv2.putText(image, body_language_class, coords,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Get status box
        cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

        # Display Class
        cv2.putText(image, 'CLASS', (95, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, body_language_class.split(' ')[
                    0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display Probability
        cv2.putText(image, 'PROB', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (
            10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    image = simplejpeg.encode_jpeg(image)
    base64_image = base64.b64encode(image).decode('utf-8')
    base64_src = f"data:image/jpg;base64,{base64_image}"
    emit('prediction', base64_src)


if __name__ == "__main__":
    CORS(app)
    socketio.run(app, ssl_context="adhoc")
