import numpy as np
import csv

import config
from feature_extraction import get_features_from_signal


def add_features_to_matrix(matrix, feat):
    if matrix.shape == (0,): # first channels with seizure
        matrix = feat
    else: 
        matrix = np.hstack((matrix, feat))
    return matrix


def create_features_and_labels(data): 
    features = []
    features_no_seizure = np.array([])
    features_seizure = np.array([])
    labels = []
    print(data.keys())
    for channel, signal in data.items() :
        print(channel, "feautre calculation")
        feat = []
        feat = get_features_from_signal(signal, feat, False) # False so the features are added after each other
        number_of_features = len(feat)
        feat = np.array([feat])
        feat = feat.reshape(number_of_features,1)

        if int(channel[-1]) == 0:
            features_no_seizure =  add_features_to_matrix(features_no_seizure, feat)
        else:
            features_seizure = add_features_to_matrix(features_seizure, feat)
    features = np.array([features])
    print("len feature no seizure ", len(features_no_seizure))
    print("len feature seizure", len(features_seizure))
    features = np.vstack((features_no_seizure, features_seizure))
    labels = [0]*number_of_features # put =0 first bc no files starts with seizure
    labels += [1]*number_of_features

    return features, labels


def remove_nans(file_with_nans):
    """
    Removing nans from files ith features because machine learning algorithms dont work with nan values.
    """

    lines = list()
    file_without_nans = config.FILE_WITHOUT_NANS

    with open(file_with_nans, 'r') as read_file:
        reader = csv.reader(read_file)
        for row_number, row in enumerate(reader, start=1):
            if "nan" not in row[4]:
                lines.append(row)

    with open(file_without_nans, 'w') as write_file:
        writer = csv.writer(write_file)
        writer.writerows(lines)

    return file_without_nans


