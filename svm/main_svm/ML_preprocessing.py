import numpy as np
from feature_extraction import get_features_from_signal


def add_features_to_matrix(matrix, feat):
    if matrix.shape == (0,): # first channels with seizure
        matrix = feat
    else: 
        matrix = np.hstack((matrix, feat))
    return matrix

def create_features_and_labels(data, find_important_channels=False, find_important_features=False): 
    if find_important_channels:
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
            if find_important_channels:
                feat = np.array([feat])
                feat = feat.reshape(number_of_features,1)

                if int(channel[-1]) == 0:
                    features_no_seizure =  add_features_to_matrix(features_no_seizure, feat)
                else:
                    features_seizure = add_features_to_matrix(features_seizure, feat)
            #elif find_important_features: 
            #    features = list(dataframe['features']) #features.append(feat)
            #    labels = list(dataframe['labels']#labels.append(int(channel[-1]))

        features = np.array([features])
        print("len feature no seizure ", len(features_no_seizure))
        print("len feature seizure", len(features_seizure))
        features = np.vstack((features_no_seizure, features_seizure))
        labels = [0]*number_of_features # put =0 first bc no files starts with seizure
        labels += [1]*number_of_features

    elif find_important_features:
        features = list(data['features'])
        labels = list(data['[label'])

 
        
    return features, labels

