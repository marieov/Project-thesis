from pyedflib import EdfReader
import pywt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- FILE READING ---------------- #

dataset_path = "C:/Users/marieov/OneDrive - NTNU/NTNU/9. semester/Project thesis/github/Project-thesis/preprocessing/CHB_MIT" # todo in config
Fs = 256 # todo: downsample to 128

# Open and store the names of all files and of the ones with seizures: 
records = open(dataset_path + '/' + 'RECORDS_TXT.txt', 'r')
records_seizures = open(dataset_path + '/' + 'RECORDS-WITH-SEIZURES_TXT.txt', 'r')
files = records.readlines()
files_seizures = records_seizures.readlines()
records.close()
records_seizures.close()
# the variable file now contains a list of all file names, on the form 'chb01/chb01_01'

dwt_no_seizure = np.array([])
dwt_seizure = np.array([])
count_s = 0
countns = 0
# ------------------ DWT ------------------ #

# for each file in the whole dataset:
for file in files:
    if (file.startswith('chb12') or file.startswith('chb22')): # for å slippe å iterere gjennom alt under debugging
        fname_path = dataset_path + '/' + file.strip()
        f = EdfReader(fname_path)
        channel_names = f.getSignalLabels() # todo: trengs denne?
        signal = f.readSignal(len(channel_names)-1)
        
        if file in files_seizures: 
            # print(file, ' has a seizure')
            # todo: finn start og slutt av seizure
            count_s += 1
            seizure_start = 1
            seizure_end = 10
            seizure = signal[seizure_start*Fs:seizure_end*Fs]
            dwt = pywt.wavedec(seizure, 'bior2.2', level=4)
            dwt_seizure = np.append(dwt_seizure, dwt)
        else: # har ikke seizure
            countns += 1
            dwt = pywt.wavedec(signal, 'bior2.2', level=4) 
            dwt_no_seizure = np.append(dwt, dwt_no_seizure)

print('Done doing dwt, calculating features')
# todo: You have to check the duration of the seizure, and use the same length for the non-seizure instances.

# ---------------- FEATURE EXTRACTION ---------------- #

# todo: use all Håkon used (snakke med Pauline + Helene)

def instantaneous_energy(data):
    """
    gives the energy distribution in log base ten for each band
    """
    if sum(i ** 2 for i in data) == 0:
        return 0  # Avoids log(0) with flat sub-bands/signals
    return np.log10((1 / float(len(data))) * sum(i ** 2 for i in data))

def teager_energy(data):
    """
    This log base ten energy operator reflects variations in both
    amplitude and frequency of the signal
    """
    sum_values = sum(abs(data[x]**2) if x == 0
                     else abs(data[x]**2 - data[x - 1] * data[x + 1])
                     for x in range(0, len(data) - 1))
    if sum_values == 0:
        return 0  # Avoids log(0) with flat sub-bands/signals
    return np.log10((1 / float(len(data))) * sum_values)

# add the features in an array

feature = []
feature_vector_no_seizure = np.array([])
feature_vector_seizure = np.array([])

for dwt_coeff in dwt_no_seizure:
    feature += [teager_energy(dwt_coeff), instantaneous_energy(dwt_coeff)] # todo: hvorfor +=?, for å legge til verdier
    feature_vector_no_seizure = np.append(feature_vector_no_seizure, feature)
print('Done finding features for no seizure (1/2)')
for dwt_coeff in dwt_seizure:
    feature += [teager_energy(dwt_coeff), instantaneous_energy(dwt_coeff)] 
    feature_vector_seizure = np.append(feature_vector_seizure, feature)
print('Done finding features for seizures (2/2)')
print(feature_vector_no_seizure.shape)
print(feature_vector_seizure.shape)

labels_no_seizure = np.zeros(len(feature_vector_no_seizure))
labels_seizure = np.ones(len(feature_vector_seizure))

labels = np.append(labels_no_seizure, labels_seizure)
features = np.append(feature_vector_no_seizure, feature_vector_seizure)

# ------------------ RANDOM FOREST ------------------ #

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


def rand_forest(feat_data, tags):
    x_train, x_test, y_train, y_test = train_test_split(feat_data, tags, test_size=0.33, random_state=42)
    forest = RandomForestClassifier(random_state=0)
    forest.fit(x_train, y_train)
    predicted_probas = forest.predict_proba(x_test)
    predictions = forest.predict(x_test)

    print(list(predictions))
    print(list(y_test))
    
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    
    #print(errors)
    #print('Error precentage:', round(np.sum(errors)/len(errors)*100, 2))
    
    #_acc,_fscore,_precision,_recall = get_metrics(y_test,predictions,predicted_probas,cmatrix_plot=True,_print=True)
    
rand_forest(features, labels)