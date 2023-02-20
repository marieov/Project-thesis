# Project-thesis

## The code is used to do epilepsy seizure detection on the CHB-MIT database. 

### From the abstract of the thesis: 
"This work presents two approaches for epileptic seizure detection. One patient-independent and one patient-dependent approach. Feature and channel reduction was done on the patient-independent approach.

Two datasets containing electroencephalographic (EEG) signals were used, CHB-MIT Scalp EEG Database and Siena Scalp EEG Database. The first one uses 22 patients and 22 to 38 channels for recording. The latter uses 14 patients and 22 to 28 channels. Both are from PyseoNet. 

To detect epileptic seizure the EEG signals were first decomposed into different sub-bands using the Discrete Wavelet Transform (DWT). From these sub-bands sixteen different features were extracted. The obtained features were used as input for Random forest (RF), Gradient boosting (GB) and Support vector machine (SVM) in order to classify seizure and seizure-free periods.

For the patient-independent approach, the feature importance for each machine learning method was found, and the most important features were chosen. These were used when finding the channel importance. The performance of the three algorithms using a low number of features and an increasing number of channels is presented.

A clear conclusion on which machine learning method, features and channels gives the best performance is not presented due to variations in results for each run of the method. The standard deviation of the accuracy was about \pm 3%. Nevertheless, high-performance measures both for a patient-dependent and patient-independent approach are presented. An accuracy between 95.9% and 100% was obtained for the patient-dependent approach, depending on which machine learning method was used. An accuracy of 97.6%, 96.4% and 88.4% were obtained for the patient-independent approach using 1-3 features and one channel, also depending on which machine learning method is used." 

### Notes about the dataset: 
- There is an error in RECORDS_WITH_SEIZURES, where there is no seizure in chb07/chb07_18.edf, but rather in chb07/chb07_19.edf. This has been corrected in RECORDS_WITH_SEIZURES_TXT.
- For some patients (e.g. patient 12) two signals (e.g. from channel 21 og 28) is from the same electrode, the euclidian distance is 0.
- In the summary.txt for patient 24 only the files with seizures are listed.
- Patient 18 inactive channels are marked with . Instead of - in the chb18-summary file.
- Some "0" are changed with "O" in the names of the channels in the summary.txt files.
- records.txt is a txt file copy of records.file.

### Disclaimer: 
The code is not tidied up because I will use the same code base as a starting point for my master thesis in spring 2023. 
