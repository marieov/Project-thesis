import numpy as np
import json
import os
import pandas as pd
from collections import Counter

import config
from ML_evaluation import evaluate_model
from ML_models import classification
from ML_preprocessing import remove_nans
from plotting import plot_results
from build_matrix import create_feature_matrix


def load_data(feat_file):
    # Load data: features, labels and channel names from .csv file
    datasamples =  pd.read_csv(remove_nans(feat_file))
    feature_importance_matrix = []
    labels = []
    channel_names = []

    print("-----------------")
    print(datasamples)
    print(Counter(datasamples['label']))

    for index in range(len(datasamples)):
        feature = datasamples['features'].iloc[index].strip('][').split(', ')

        feature_importance_matrix.append(feature)
        labels.append(datasamples['label'].iloc[index])
        channel_names.append(datasamples['channel'].iloc[index])
    return feature_importance_matrix, labels, channel_names


def load_importances(importance_file):
    importances = json.load(open(importance_file,'r'))
    importances_decending = dict(sorted(importances.items(), key=lambda x:abs(x[1]), reverse=True)) # abs value bc for svm the importances are both negative and positive 
    return importances_decending


def find_importance(feat_file, importances_file, names, find_important_channels, find_important_features, mdi, mda):
    if not (os.path.exists(feat_file)): # saving variables to file so the runtime is shorter
        create_feature_matrix(feat_file, find_important_channels, find_important_features) # for finding feature importances

    # Load data
    if find_important_features:
        importance_matrix, labels, channel_names = load_data(feat_file)
    else: # find important channels
        # load_data(feat_file) # TODO: generalize to use til, not the 4 lines below
        importance_matrix = np.loadtxt(feat_file) 
        labels = np.loadtxt(config.LABELS_FILE)
        channel_names = np.loadtxt(config.CHANNELS_FILE, dtype=str)
        names = channel_names

    # Machine learning and evaluation
    if find_important_features:
        print("Finding most important features (over all channels) for", config.MODEL)
    else: 
        print("Finding most important channels for", config.MODEL)
    fitted_model, predicted, X_test, y_test = classification(config.MODEL, importance_matrix, labels)
    evaluate_model(fitted_model, predicted, names, X_test, y_test, find_important_channels, find_important_features, mdi, mda)
    plot_results(importances_file, y_test, predicted, find_important_features)

