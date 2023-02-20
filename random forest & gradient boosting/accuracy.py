import os 
import pandas as pd
import numpy as np

import config
from plotting import plot_accuracies
from importances import load_importances, remove_nans
from ML_evaluation import evaluate_model
from ML_models import classification

def calculate_accuracy(find_important_channels=False, find_important_features=True, mdi=False, mda=True):
    '''
    Find the most important channels and their importance
    Change to finding accruacy for feature importance 
    Add one and one channel from channel importances (sorted) and run feature extraction
    Plot the resulting accuracy with an increasing number of channels, depending on their importance
    '''
    print("Calculating performance with increasnig number of chnnels (based on channels sorted by decreasing importance)")
    channel_names = np.loadtxt(config.CHANNELS_FILE, dtype=str)
    importances_decending = load_importances(config.IMPORTANCE_FILE_CHANNELS)

    if not (os.path.exists(config.ACCURACY_FILE)): 
            
        print("---------- CHANNEL names ", channel_names)
        datasamples =  pd.read_csv(remove_nans(config.FEAT_FILE_FEATURES))
        channels_sorted = [channel[:-2] for channel in channel_names if channel[-1]=="0"] # the channel names with duplicates
        print("----------CHANNELS SORTED ", channels_sorted)
        print("Calculating accuracies") 

        accuracies = []
        labels = np.array([])
        important_features = np.array([])

        for channel in importances_decending.keys():
            # henter ut alle rader i df med den channelen
            # må konvertere fra channel no til channel_name

            print("Channel no and name:", int(channel), channels_sorted[int(channel)])

            channel_name = channels_sorted[int(channel)] # getting the name of te channel by fetchine the name in the channels_sorted list by the index
            # whish to use the channel to fetch the right row in the feature importance matrix 

            for index, ch in enumerate(datasamples['channel']): 
                if ch == channel_name: # get the row (channel) with the features from that channel MEN DENNE ER 3 OG P7
                    feature = datasamples['features'].iloc[index].strip('][').split(', ')

                    if len(important_features) == 0: 
                        important_features = np.array(feature)
                    else:
                        important_features = np.vstack((important_features, feature))
                    labels = np.append(labels, datasamples.loc[index]['label'])

            fitted_model, predicted, X_test, y_test = classification(config.MODEL, important_features, labels)
            accuracy = evaluate_model(fitted_model, predicted, channel_names, X_test, y_test, find_important_channels, find_important_features, mdi, mda)

            # Collecting all accuracies to plot them:
            accuracies = np.append(accuracies,accuracy)
            np.savetxt(config.ACCURACY_FILE, accuracies)
    plot_accuracies(importances_decending) # TODO: er denne parameteren nødvendig? Den lagres vel til fil den eog?
    
