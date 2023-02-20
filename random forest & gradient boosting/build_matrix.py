import pandas as pd
import numpy as np

import config
from file_reading import read_record_files
from get_signals import get_signal_from_files
from ML_preprocessing import create_features_and_labels


def save_feature_matrix(df, feat_file):
    df.to_csv(feat_file)


def create_feature_matrix(feat_file, find_important_channels, find_important_features):
    
    # Crating a dict of channels and their signals
    print("Extracting features and labels from signal")
    channel_signals = {} # collecting the signlas on each channel
    channel_names = []

    df_feature = pd.DataFrame(columns = ['channel', 'trail', 'label', 'features'])

    files, files_seizures = read_record_files()

    for i in range(config.START_PATIENT, (config.END_PATIENT)+1):   
        patient_number = str(i).zfill(2)
        print("Patient no", patient_number)
        if find_important_channels:
            channel_signals,df_feature = get_signal_from_files(files, files_seizures, patient_number, channel_signals, df_feature, find_important_features)            
            for channel, signal in channel_signals.items() : # debugging
                print (channel, '\t', len(signal))
            # not saving dict to file bc that takes ages, the signals are really long!
            channel_names = list(channel_signals.keys()) # FC1_0, P7_0, FC1_1, P7_1
            features, labels = create_features_and_labels(channel_signals)
            print("----------", len(channel_names), len(features), len(labels))

            #dict = {'channel': channel_names, 'label': labels, 'freatures': features} 
            
            np.savetxt(feat_file, features)
            np.savetxt(config.LABELS_FILE, labels)
            np.savetxt(config.CHANNELS_FILE, channel_names, fmt="%s")
            
            print(type(channel_names),  list(channel_signals.keys()))
            #save_feature_matrix(df_channel, feat_file)  

        elif find_important_features:
            channel_signals,df_feature = get_signal_from_files(files, files_seizures, patient_number, channel_signals, df_feature, find_important_features)
            for index, channel in enumerate(list(df_feature['channel'])):
                if channel + '_' + str(list(df_feature['label'])[index]) not in channel_names:
                    channel_names.append(channel + '_' + str(list(df_feature['label'])[index]))

            print(df_feature.head())
            save_feature_matrix(df_feature, feat_file)
