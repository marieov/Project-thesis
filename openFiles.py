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

# ----------------  ---------------- #

#dwt_no_seizure = np.array([])
dwt_seizure = []
channelss = 0 # debugging
trailss = 0 # debugging

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

# for each file in the whole dataset:
# the variable file now contains a list of all file names,
# on the form 'chb01/chb01_01'
feature_vector = []
for file in files:
    if (file.startswith('chb12')): # for å slippe å iterere gjennom alt under debugging
        # skal være 24 trails i 12
        fname_path = dataset_path + '/' + file.strip()
        f = EdfReader(fname_path)
        channel_names = f.getSignalLabels() # todo: en bedre løsning?
        trailss += 1 # debugging
        channelss = 0 # debugging

        for channel in range(len(channel_names)): # 28 eller 29 i 12
            channelss += 1
            signal = f.readSignal(channel)
            dwt = pywt.wavedec(signal, 'bior2.2', level=4)
            # dwt_seizure.append(dwt) # ikke nødvendig, 
            # må da itereres over etterpå 
            feature = []
            for dwt_value in dwt: 
                feature += [teager_energy(dwt_value), instantaneous_energy(dwt_value)]
            feature_vector += feature
            print(len(feature_vector))
           
        #print('channels', channelss, 'trails', trailss, 'len av dwt vektor', len(dwt_seizure)) # burde være 

# todo: seizure / no seizure 
# todo: labels y