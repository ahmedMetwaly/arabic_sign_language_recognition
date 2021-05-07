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
import os
from matplotlib import pyplot as plt
import uuid
import time
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
@app.home('/home')
def home():
  return "hello"

if __name__ == "__main__":
    app.secret_key = 'ItIsASecret'
    app.debug = True
    app.run()
