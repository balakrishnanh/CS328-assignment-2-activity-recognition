# -- IMPORTS START --
import pandas as pd
import glob
import re
import os
import sys
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split
# -- IMPORTS END --

# enable zooming into graphs
plt.rcParams['figure.figsize'] = [9, 6] # width, height in inches

# Helper function to visualize model - Do not modify
def viz_tree(dt_model,features_frames,cnames,max_depth=3):
    # Fix feature names as list
    feature_names = features_frames.columns.tolist()

    # Only the top `max_depth` levels are drawn so the boxes stay readable
    fig, ax = plt.subplots(figsize=(20,8))
    tree.plot_tree(dt_model,
                   feature_names=feature_names,
                   fontsize=8,
                   class_names=cnames,
                   filled=True,
                   max_depth=max_depth,
                   ax=ax)

    plt.title('Decision Tree')
    plt.savefig('dt.png')

#Do not modify
def calc_magnitude(data):

    # Calculate magnitude
    data['accel_mag'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2) # absolute accel magnitude
    data['accel_mag'] = data['accel_mag'] - data['accel_mag'].mean() # detrend: "remove gravity"

    return data

#Do not modify
def remove_noise(data,sampling_rate):
    from scipy.signal import butter, filtfilt, find_peaks

    # Low pass filter
    cutoff = 5 # Hz
    order = 2
    b, a = butter(order, cutoff/(sampling_rate/2), btype='lowpass')
    data['filtered_accel_mag'] = filtfilt(b, a, data['accel_mag'])

    return data


#Do not modify
def add_features(window):
    features = {}
    features['avg'] = window['filtered_accel_mag'].mean()
    features['max'] = window['filtered_accel_mag'].quantile(1)
    features['med'] = window['filtered_accel_mag'].quantile(0.5)
    features['min'] = window['filtered_accel_mag'].quantile(0)
    features['q25'] = window['filtered_accel_mag'].quantile(0.25)
    features['q75'] = window['filtered_accel_mag'].quantile(0.75)
    features['std'] = window['filtered_accel_mag'].std()
    df = pd.DataFrame([features])
    return df




