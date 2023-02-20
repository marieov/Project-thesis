import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split 
import time

import config
from file_reading import read_record_files
from get_signals import get_signal_from_files
from ML_evaluation import evaluate_model, plot_accuracies
from ML_models import random_forest, support_vector_machine, gradient_boosting
from ML_preprocessing import create_features_and_labels

from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action='ignore', category=FutureWarning)


# -------- Variables -------- #

mdi = False # use mean decrease in impurity eller mean decreasy in accuacy
mda = not(mdi) 

feat_file_features = "features_f_no_nans.csv"

feat_file_channels = "features_c.txt"
labels_file_channels = "labels_c.txt"
channels_file_channels = "channel_names_c.txt"

feature_names = ["std", "rms", "katz_fractal_dim"] 

#["var", "ptp_amp", "kurtosis", "instantaneous_energy", "teager_energy", "higuchi_fractal_dimension","petrosian_fractal_dimension", "std", "skewness", 
#"line_length", "rms", "hjort_mobility", "hjort_complexitY", "katz_fractal_dim", "sevcik_fraction_dim","mean"] # names needed for plotting

#models = ["rf", "gb", "svm"]
model = "svm"


# -------- Creating features and labels -------- #
# Creating one matrix for finding feature importances and one for finding channel importances (see description in thesis)

def create_feature_matrix(feat_file, find_important_channels, find_important_features, labels_file='', channels_file=''):
    
    # Crating a dict of channels and their signals
    print("Extracting features and labels from signal")
    channel_signals = {} # collecting the signlas on each channel
    channel_names = []
    
    datasamples = pd.DataFrame(columns = ['channel', 'trail', 'label', 'features'])


    files, files_seizures = read_record_files()

    for i in range(config.start_patient, (config.end_patient)+1):   
        patient_number = str(i).zfill(2)
        print("Patient no", patient_number)
        if find_important_channels:
            channel_signals,datasamples = get_signal_from_files(files, files_seizures, patient_number, channel_signals, datasamples, find_important_features)            
            for channel, signal in channel_signals.items() : # debugging
                print (channel, '\t', len(signal))
            # not saving dict to file bc that takes ages, the signals are really long!
            channel_names = list(channel_signals.keys()) # FC1_0, P7_0, FC1_1, P7_1
            features, labels = create_features_and_labels(channel_signals, find_important_channels, find_important_features )
            # lagre disse til file
                        
            np.savetxt(feat_file, features)
            np.savetxt(labels_file, labels)
            np.savetxt(channels_file, channel_names, fmt="%s")

            print(type(channel_names),  list(channel_signals.keys()))

        elif find_important_features:
            channel_signals,datasamples = get_signal_from_files(files, files_seizures, patient_number, channel_signals, datasamples, find_important_features)
            for index, channel in enumerate(list(datasamples['channel'])):
                if channel + '_' + str(list(datasamples['label'])[index]) not in channel_names:
                    channel_names.append(channel + '_' + str(list(datasamples['label'])[index]))
            #features, labels = create_features_and_labels(datasamples, find_important_channels, find_important_features )
            print(datasamples.head())
            datasamples.to_csv(feat_file)

# ------------- MACHINE LEARNING -----------#
def classification(model, features, labels, names): # TODO dårlig navn
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
    if find_important_channels:
        file_name = model + "_channels.pdf"
    elif find_important_features:
        file_name = model + "_features.pdf"
    start = time.time()
    if model == "rf": 
        ml_model, predicted = random_forest(X_train, X_test, y_train, y_test)
    elif model == "gb":
        ml_model, predicted = gradient_boosting(config.learning_rate, X_train, X_test, y_train, y_test)
    elif model =="svm":
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        ml_model, predicted = support_vector_machine(X_train, X_test, y_train, y_test)
    end = time.time()
    print("ELAPSED TIME FOR ML MODEL", end - start)
    importances, accuracy = evaluate_model(file_name, ml_model, predicted, names, X_test, y_test, y_train, features, labels, find_important_channels, find_important_features, mdi, mda)
    # TODO hva sendes inn som find iortant channels og features her nå og hvorfan påvirker d programmer?
    # TODO plot importances as a separate line

    importances_decending = dict(sorted(importances.items(), key=lambda x:abs(x[1]), reverse=True)) # abs value bc for svm the importances are both negative and positive 
    if find_important_channels:
        print("Most important channels for ", model, ": ", list(importances_decending.keys()))
    elif find_important_features:
        print("Most important features for ", model, ": ", list(importances_decending.keys()))

    return importances_decending, accuracy



# -------- Creating feature importnace matrix -------- #
find_important_channels=False
find_important_features=True
if not (os.path.exists(feat_file_features)): # saving variables to file so the runtime is shorter
    create_feature_matrix(feat_file_features, find_important_channels, find_important_features) # for finding feature importances
# KJØRING AV KODEN FORDRER AT DE FEATURENE MAN VIL HA ALLEREDE ER VALGT (de finner man etter å ha kjørt koden med importances, _ = evalutes_models og find important features = True)
 
datasamples = pd.read_csv(feat_file_features)
feature_importance_matrix = []
labels = []
channel_names = []

# Get lists of features, labels and channel names from .csv file
print("-----------------")
print(datasamples)
for index in range(len(datasamples)):
    feature = datasamples['features'].iloc[index].strip('][').split(', ')

    feature_importance_matrix.append(feature)
    labels.append(datasamples['label'].iloc[index])
    channel_names.append(datasamples['channel'].iloc[index])

print("Finding most important features (over all channels) for", model)
importances_decending, _ = classification(model, feature_importance_matrix, labels, feature_names)

# -------- Create the channel importance matrix -------- #

find_important_channels=True
find_important_features=False
if not (os.path.exists(feat_file_channels)):
    create_feature_matrix(feat_file_channels, find_important_channels, find_important_features, labels_file_channels, channels_file_channels) # for finding channel importances

channel_importance_matrix = np.loadtxt(feat_file_channels) 
labels = np.loadtxt(labels_file_channels)
channel_names = np.loadtxt(channels_file_channels, dtype=str)

print("Finding most important channels for", model)
importances_decending,_ = classification(model, channel_importance_matrix, labels, channel_names)

# -------- Channels accumulating -------- #

find_important_channels=False
find_important_features=True
print("Calculating performance with increasnig number of chnnels (based on channels sorted by decreasing importance")

# FØrst finne de viktigste channels med channell importance, så bytte og finne accuracy for feature importance setup
# Legge til en og en channel fra channel imprtoances og kjøre feature extraction og plotte accuracy
# Increasing number of channels, depending on their importance

channels_sorted = [channel[:-2] for channel in channel_names if channel[-1]=="0"] # listing all channels by names, with duplicates
# TODO: are these names found in ML_evaluation too, evaluate_model?

if not (os.path.exists("acc_" + str(model) + ".pdf")): 
    print("Calculating accuracies")
    accuracies = []
    labels = np.array([])
    important_features = np.array([])
    for channel in importances_decending.keys(): # channel is an int # TODO: DENNE ER FRA DEN SISTE I MODELS, så det blir svm hver gang
        # henter ut alle rader i df med den channelen
        # må konvertere fra channel no til channel_name

        """
        # finding the most importnat channels and reoing rows from the channel importansces setup
        if 'important_features' not in locals(): # important channels stacked horizontally
            important_features = np.hstack((np.array([]), feature_importance_matrix[:, channel])) # get the column with that value
            # er dette egt rikitg? FOrdi hva om channels kommer sånn: ch1_0, ch2_0, ch1_1, ch2_1, ch3_0, ch3_1, da blir importance decending 2, 3, 1 feks 
            # bruke channel_names instead
        else:
            important_features = np.hstack((important_features, feature_importance_matrix[:, channel].reshape(len(feature_importance_matrix), 1)))
        important_features = important_features.reshape(len(feature_importance_matrix), -1)
        # labels are the same
        """
        print("Channel no and name:", channel, channels_sorted[channel])

        channel_name = channels_sorted[channel] # getting the name of te channel by fetchine the name in the channels_sorted list by the index
        # whish to use the channel to fetch the right row in the feature importance matrix 

        # get the index (of the channel we want to fetch features from) in the features_importance_matrix (which is the same as for channel_signal) 
        #channel_index = (list(channel_names).index(channel_name + "_0"), list(channel_names).index(channel_name + "_1")) # the index in the feature_importance_matrix, both for seizure free nad seizure signal

        #channel_signals = get_signal_from_files(files, files_seizures, patient_number, channel_signals)
        for index, ch in enumerate(datasamples['channel']): 
            if ch == channel_name: # get the row (channel) with the features from that channel MEN DENNE ER 3 OG P7
                feature = datasamples['features'].iloc[index].strip('][').split(', ')

                if len(important_features) == 0: 
                    important_features = np.array(feature)
                else:
                    important_features = np.vstack((important_features, feature))
                labels = np.append(labels, datasamples.loc[index]['label'])


        _, accuracy = classification(model, important_features, labels, feature_names)

        # Collecting all accuracies to plot them:
        accuracies = np.append(accuracies,accuracy)
            
        #important_features = important_features.reshape(-1, len(feature_importance_matrix))
        
        # running the ML with the rows from the n most important channels
        """X_train, X_test, y_train, y_test = train_test_split(important_features, labels, test_size=0.33, random_state=42)

        file_name = model + str(channel) + ".pdf"
        if rf:
            rf, predicted = random_forest(X_train, X_test, y_train, y_test)
        elif gb:
            gb, predicted = gradient_boosting(X_train, X_test, y_train, y_test)
        elif svm:
            svmClf, predicted = support_vector_machine(X_train, X_test, y_train, y_test)
        _, accuracy = evaluate_model(file_name, model, predicted, channel_names, feature_names, X_test, y_test, y_train, feature_importance_matrix, labels, find_important_channels, find_important_features, mdi, mda)###
        """


        np.savetxt("acc_" + str(model) + ".txt", accuracies) # TODO put outside for loop?
    plot_accuracies(accuracies, importances_decending)
else: 
    print("Accuracies already saved, reading from file")
    accuracies = np.loadtxt("acc_" + str(model) + ".txt")
    plot_accuracies(accuracies, importances_decending)

