from pyedflib import EdfReader
import pywt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
import re

# ---------------- FILE READING ---------------- #

dataset_path = "C:/Users/marieov/OneDrive - NTNU/NTNU/9. semester/Project thesis/github/Project-thesis/CHB_MIT" # todo in config
Fs = 256 # todo: downsample to 128

# Open and store the names of all files and of the ones with seizures: 
records = open(dataset_path + '/' + 'RECORDS_TXT.txt', 'r')
records_seizures = open(dataset_path + '/' + 'RECORDS-WITH-SEIZURES_TXT.txt', 'r')
files = records.readlines()
files_seizures = records_seizures.readlines()
records.close()
records_seizures.close()
# the variable file now contains a list of all file names, on the form 'chb01/chb01_01'

# ---------------- FEATURES ---------------- #

def teager_energy(data):
    sum_values = sum(abs(data[x]**2) if x == 0
                     else abs(data[x]**2 - data[x - 1] * data[x + 1])
                     for x in range(0, len(data) - 1))
    if sum_values == 0:
        return 0  # Avoids log(0) with flat sub-bands/signals
    return np.log10((1 / float(len(data))) * sum_values)

def instantaneous_energy(data):
    if sum(i ** 2 for i in data) == 0:
        return 0  # Avoids log(0) with flat sub-bands/signals
    return np.log10((1 / float(len(data))) * sum(i ** 2 for i in data))

# ---------------- DWT AND FEATURE EXTRACTION ---------------- #

def get_seizure_start_end(file_path): 
    start_time = []
    end_time = []
    summary_file = open(fname_path[:-7] + '-summary.txt','r')
    for line in summary_file: 
        seizure_start = re.search("Seizure\s\d\sStart", line)
        if seizure_start is not None: 
            start_time += line.split()[4:][:-1]
            # obs det er np en string
        seizure_end = re.search("Seizure\s\d\sEnd", line)
        if seizure_end is not None: 
            end_time += line.split()[4:][:-1]
    summary_file.close()
    return start_time, end_time

#dwt_no_seizure = np.array([])
dwt_seizure = []
channelss = 0 # debugging
trailss = 0 # debugging

# for each file in the whole dataset:
# the variable file now contains a list of all file names,
# on the form 'chb01/chb01_01'
features_no_seizure = []
features_seizure = []
Fs = 256

for file in files:
    if (file.startswith('chb12/')): # for å slippe å iterere gjennom alt under debugging
        # skal være 24 trails i 12
        fname_path = dataset_path + '/' + file.strip()
        f = EdfReader(fname_path)
        channel_names = f.getSignalLabels() # todo: en bedre løsning?
        trailss += 1 # debugging
        channelss = 0 # debugging

        for channel in range(len(channel_names)): # 28 eller 29 i 12
            channelss += 1
            signal = f.readSignal(channel)

            if file in files_seizures: # seizure
                """
                Hente ut riktige sekunder
                Kan være flere seizures i en fil: 
                                 
                Lese "Seizure %arbitatray number% Start time" fra summary
                Lese "Seizure %arbitatray number% End time" fra summary
                Legge i start_times = [] og end_times = []
                for time in start_time: 
                  signal, dwt, feature...


                poppe fra signal så kan jeg bruke resten til no_seizure
                ta vekk elsen, fordi da er jo seizure borte uansett
                MEN da kan det ikke brukes fordi indexene er forskøvet
                """
                # Get seizures from files: 
                start_time, end_time = get_seizure_start_end(fname_path)

                for i in range(len(start_time)): 
                    seizure_start = start_time[i]
                    seizure_end = end_time[i]
                    seizure_signal = signal[int(seizure_start*Fs):int(seizure_end*Fs)]

                    dwt = pywt.wavedec(signal, 'bior2.2', level=4)
                    # dwt_seizure.append(dwt) # ikke nødvendig, 
                    # må da itereres over etterpå 
                    feature = []
                    for dwt_value in dwt: 
                        features_seizure += [teager_energy(dwt_value), instantaneous_energy(dwt_value)]
                    features_seizure += feature
                    print('Features_seizure', len(features_seizure)) # denne blir lenger nå som man kan ha flere seizures i samme fil
                        

            else: # no seizure
                dwt = pywt.wavedec(signal, 'bior2.2', level=4)
                feature = []
                for dwt_value in dwt: 
                    features_no_seizure += [teager_energy(dwt_value), instantaneous_energy(dwt_value)]
                features_no_seizure += feature
                print(len('features_no_seizure', features_no_seizure))
           
        #print('channels', channelss, 'trails', trailss, 'len av dwt vektor', len(dwt_seizure)) # burde være 

labels_no_seizure = np.zeros(len(features_no_seizure))
labels_seizure = np.ones(len(features_seizure))

labels = np.append(labels_no_seizure, labels_seizure)
features = np.append(features_no_seizure, features_seizure)


# ------------------ RANDOM FOREST ------------------ #

"""
def confusion_matrix_heatmap(target, predicted, perc=False):
    plt.figure()
    data = {'y_Actual': target, 'y_Predicted': predicted}
    df = pd.DataFrame(data, columns=['y_Predicted','y_Actual'])
    c_matrix_l = pd.crosstab(df['y_Predicted'], df['y_Actual'],
        rownames=['Predicted'], colnames=['Actual'])
    if perc:
        sns.heatmap(c_matrix_l/np.sum(c_matrix_l),
            annot=True, fmt='.2%', cmap='Blues')
    else:
        sns.heatmap(c_matrix_l, annot=True, fmt='d')


def get_metrics(target, predicted,predicted_prob=None, cmatrix_plot=False, _print=False, _title="", _average="macro"):#micro, macro
    if cmatrix_plot:
        confusion_matrix_heatmap(target, predicted)#, _title=_title
    _acc = round(accuracy_score(target, predicted), 3)
    _fscore = round(f1_score(target, predicted, average=_average), 3)
    _precision = round(precision_score(target, predicted, average=_average), 3)
    _recall = round(recall_score(target, predicted, average=_average), 3)
    
    if _print:

        print( "acc: ",_acc, "fscore: ", _fscore, "precision: ", _precision, "recall: ", _recall)
    return _acc,_fscore,_precision,_recall
"""

def rand_forest(feat_data, tags):
    print('Random forest')
    x_train, x_test, y_train, y_test = train_test_split(feat_data, tags, test_size=0.33, random_state=42)
    
    # x_train should be a 2D features matrix with shape (n_instances, n_features)
    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)
    rf = RandomForestClassifier(random_state=0)
    
    rf.fit(x_train, y_train)

    y_rf_predict_test = rf.predict(x_test)
    y_rf_predict_train = rf.predict(x_train)
    RF_train = accuracy_score(y_train, y_rf_predict_train)
    RF_test = accuracy_score(y_test, y_rf_predict_test)
    important_features_dict = {}
    for xx, i in enumerate(rf.feature_importances_):
        important_features_dict[xx] = i

    important_features_list = sorted(important_features_dict, key=important_features_dict.get, reverse = True)

    print('Most immportant features', important_features_list)

    predicted_probas = rf.predict_proba(x_test)
    predictions = rf.predict(x_test)

    print(list(predictions))
    print(list(y_test))
    
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    
    #print(errors)
    print('Error precentage:', round(np.sum(errors)/len(errors)*100, 2))
    
    #_acc,_fscore,_precision,_recall = get_metrics(y_test,predictions,predicted_probas,cmatrix_plot=True,_print=True)
    

# bruke en og en feature?
rand_forest(features, labels)