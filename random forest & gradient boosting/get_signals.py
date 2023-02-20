import config 
import re
from pyedflib import EdfReader
import numpy as np
import re

from feature_extraction import get_features_from_signal


def add_signal_to_channel(signal, channel, channel_signals, seizure): 
    '''
        Adding the signal to the channel

        param signal: signal
        param channel: channel name
        param channel_signals: dict to add signals to corresponding channel, dict
        param seizure: True if the signal contain seizure, False if the signal does not contain seizure, bool

        return channel_signals: dict after adding signals to corresponding channel, dict
        return signal: signal
        return seizure: True if the signal contain seizure, False if the signal does not contain seizure, bool
    '''
    if not all((s == 0.0 or s == signal[0]) for s in signal) and channel not in config.UNWANTED_CHANNELS: # channel not in use
        # this is not optimal (with adding the seizure as string, should be a dataframe), changing this later
        if channel + '_' + str(int(seizure)) not in channel_signals:
             channel_signals[channel + '_' + str(int(seizure))] = signal
        else:               
            channel_signals[channel + '_' + str(int(seizure))] += signal

    return channel_signals, signal, seizure


def get_seizure_start_end(file): 
    '''
        Finds the start time and end time for the seizure from the summary file for each patient. 
        There can be more than one seizure per file

        param file: name of the file, string

        return start_time: when the seizures in the file starts, list of strings  
        return end_time: when the seizures in the file ends, list of strings
    '''
    start_time = []
    end_time = []
    correct_file = False
    summary_file = open(config.DATASET_PATH + '/' + file[:5] + '/' + file[:5] + '-summary.txt','r')

    for line in summary_file:
        if line.startswith("File Name:"):
            # given that the line is the one with the name of the file, 
            # flag if it is the correct file
            # using a flag because mulitple lines are read in order to collect
            # all start and end times
            if line.strip() == ("File Name: " + file[6:]).strip():
                correct_file = True 
            else: 
                correct_file = False
                
        if correct_file:  
            seizure_start = re.search("Seizure\s(\d\s)?Start", line) # some files are on the format "Seizure 1 Start", some just "Seizure Start" 
            if seizure_start is not None: 
                # retrieve only the part with the number from the line (after the colon)
                start = re.search(":(\s)*\d+", line).group(0)
                start_time = np.append(start_time, start[2:])
            
            seizure_end = re.search("Seizure\s(\d\s)?End", line)
            if seizure_end is not None: 
                # retrieve only the part with the number from the line (after the colon)
                end = re.search(":(\s)*\d+", line).group(0)
                end_time = np.append(end_time, end[2:])
    
    summary_file.close()
    return start_time, end_time


def get_signal_start_interval(signal, channel, channel_signals, start_time, interval):
    seizure = False
    
    seizure_start = int(start_time[interval]) # in seconds                    
    seizure_free_signal = signal[:int(seizure_start*config.FS)]

    channel_signals, signal_interval, seizure = add_signal_to_channel(seizure_free_signal, channel, channel_signals, seizure)
    return channel_signals, signal_interval, seizure
        
    
def get_signal_seizure_interval(signal, channel, channel_signals, start_time, end_time, interval):
    seizure = True
    
    seizure_start = int(start_time[(interval-1)//2]) # in seconds
    seizure_end = int(end_time[(interval-1)//2]) # in seconds
    
    seizure_signal = signal[(int(seizure_start)*config.FS):int(seizure_end*config.FS)]
      
    channel_signals, signal_interval, seizure = add_signal_to_channel(seizure_signal, channel, channel_signals, seizure)
    
    return channel_signals, signal_interval, seizure


def get_signal_intermediate_interval(signal, channel, channel_signals, start_time, end_time, interval):
    seizure = False
    
    seizure_start = int(start_time[interval//2])
    seizure_end = int(end_time[(interval-2)//2]) 
    seizure_free_signal = signal[(seizure_end*config.FS):int(seizure_start*config.FS)]

    channel_signals, signal_interval, seizure = add_signal_to_channel(seizure_free_signal, channel, channel_signals, seizure)
    return channel_signals, signal_interval, seizure


def get_signal_last_interval(signal, channel, channel_signals, end_time, interval):
    seizure = False
    
    seizure_end = int(end_time[(interval-2)//2]) # in seconds, -2 because we are looking at the last end time, not the current
    seizure_free_signal = signal[int(seizure_end*config.FS):]

    channel_signals, signal_interval, seizure = add_signal_to_channel(seizure_free_signal, channel, channel_signals, seizure)
    return channel_signals, signal_interval, seizure


def get_signal_from_interval(signal, channel, channel_signals, start_time, end_time, interval):
    
    if interval == 0: # start of the signal
        channel_signals, signal_interval, seizure = get_signal_start_interval(signal, channel, channel_signals, start_time, interval)
       
    elif (interval % 2) != 0: # seizure
        channel_signals, signal_interval, seizure = get_signal_seizure_interval(signal, channel, channel_signals, start_time, end_time, interval)
    
    else: 
        number_of_seizures = len(start_time)
        if interval == (number_of_seizures*2): #last part
            channel_signals, signal_interval, seizure = get_signal_last_interval(signal, channel, channel_signals, end_time, interval)

        else: #intermediate part
            channel_signals, signal_interval, seizure = get_signal_intermediate_interval(signal, channel, channel_signals, start_time, end_time, interval)
            
    return channel_signals, signal_interval, seizure


def get_signal_from_file(edf_file, channel_no):
    signal = edf_file.readSignal(channel_no) 
    signal = signal[::config.DOWNSAMPLING_RATE].tolist() 
    return signal


def get_signal_from_files(files, files_seizures, patient_number, channel_signals, df, find_important_features):
    '''
    Reads the signal for each trail and each channel.      
    # TODO: If the file has an epileptic seizure the seizure is splitted into 6 sec segments. 
    Each signal is decomposed into levels using DWT. 
    From the four levels features are found.
    The features are added in a dataframe. 
    The rows in the dataframe are the features from the different parts of the signal (seizure/no seizure)
    
        param files: names of all the files, on the form 'chb01/chb01_01', list
        param files_seizures: names of all the files with seizures, on the form 'chb01/chb01_01', list
        param patient_number: patient number, int

        return dataset: dataframe where the data (features, patient, label) is added
    '''
    
    # Open the file and read the channels:
    for file in files:
        if file.startswith('chb' + str(patient_number)): # this is added to make it easier to run for fewer patients 
            fname_path = config.DATASET_PATH + '/' + file.strip()
            print('File', file) # debugging
            
            edf_file = EdfReader(fname_path)
            
            channel_names =  edf_file.getSignalLabels() # all channels

            # If the file has a seizure we need to separate the seizure signal from the non-seizure signal:
            if file in files_seizures: # If the file has a seizure we need to separate the seizure signal from the non-seizure signal
                
                # Finding the start and end time of the seizures (lists)
                start_time, end_time = get_seizure_start_end(file.strip())
                # For each seizure we collect the signal to do dwt and find features
                number_of_seizures = len(start_time)
                intervals = number_of_seizures*2+1 # number of seizures/no seizure intervals in file. E.g. if there is one seizure, there are 3 intervals.
                
                for interval in range(intervals):

                    print('Interval number', interval+1, 'of', intervals) # debugging
                    
                    for channel_no,channel in enumerate(channel_names): 
                        channel = channel.split("-")[0]
                        signal = get_signal_from_file(edf_file, channel_no)
                        channel_signals, signal_interval, seizure = get_signal_from_interval(signal, channel, channel_signals, start_time, end_time, interval)
                        if find_important_features: 
                            features = get_features_from_signal(signal_interval, feature_vector=[], seizure=False) # false so features are added after eachother
                            df = df.append({'channel' : channel, 'trail' : file + str(interval), 'label' : seizure, 'features': features}, ignore_index = True)


            # No seizure in file
            else:
                for channel_no,channel in enumerate(channel_names):
                    channel = channel.split("-")[0]
                    signal = get_signal_from_file(edf_file, channel_no)
                    channel_signals, signal_interval, seizure = add_signal_to_channel(signal, channel, channel_signals, seizure = False)
                    if find_important_features:
                        if not all((s == 0.0 or s == signal[0]) for s in signal) and channel not in config.UNWANTED_CHANNELS:
                            features = get_features_from_signal(signal, feature_vector=[], seizure=False) # false so features are added after eachother
                            df = df.append({'channel' : channel, 'trail' : file + str(0), 'label' : seizure, 'features': features}, ignore_index = True)

    return channel_signals, df
