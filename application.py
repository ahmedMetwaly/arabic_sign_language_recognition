import os
import cv2
import pickle
import base64

import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit, send
from sklearn.metrics import accuracy_score  # Accuracy metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import csv
import mediapipe as mp  # Import mediapipe

from utils import decode_image_base64

app = Flask(__name__, static_url_path='')
app.config['SECRET_KEY'] = 'somethingsoamazingomg!!!'
socketio = SocketIO(app)


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
    nparr = np.frombuffer(img_binary_str, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Recolor Feed
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make Detections
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        results = holistic.process(image)
    # print(results.face_landmarks)

    # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
    # Export coordinates

    # Extract Pose landmarks
    if isInitialized(results.pose_landmarks) == False:
        emit("speak", 'move back to appear')

    # Extract left_hand landmarks
    if isInitialized(results.left_hand_landmarks) == False:
        emit("speak", "move back one step to detect left hand")

    # Exract right_hand landmarks
    if isInitialized(results.right_hand_landmarks) == False:
        emit("speak", "move back one step to detect right hand")

    # Concate rows
    if isInitialized(results.pose_landmarks) == True and isInitialized(results.right_hand_landmarks) == True and isInitialized(results.left_hand_landmarks) == True:
        # pose detection
        pose = results.pose_landmarks.landmark
        pose_row = list(np.array(
            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
        # left hand coordinates
        left_hand = results.left_hand_landmarks.landmark
        left_hand_row = list(np.array(
            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())

        # right hand coordinates
        right_hand = results.right_hand_landmarks.landmark
        right_hand_row = list(np.array(
            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())

        rowRLH = pose_row+right_hand_row+left_hand_row
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

    _, image = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(image).decode('utf-8')
    base64_src = f"data:image/jpg;base64,{base64_image}"
    emit('prediction', base64_src)


if __name__ == "__main__":
    app.config["DEBUG"] = False
    socketio.run(app,ssl_context="adhoc")
