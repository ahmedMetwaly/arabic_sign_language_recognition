import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import csv
from matplotlib import pyplot as plt
import cv2
import mediapipe as mp # Import mediapipe


app = Flask(__name__)
with open('signs4.pkl', 'rb') as f:
    model = pickle.load(f)

def isInitialized(hand):
    try:
        if hand.IsInitialized() == True:
            return True
    except: return False

@app.route('/')
@app.route('/home')
def home():
   print('hello')

if __name__ == "__main__":
    app.run(debug=True)
