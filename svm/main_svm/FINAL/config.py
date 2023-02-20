
dataset_path = "../CHB_MIT" #"C:/Users/marie/OneDrive - NTNU/NTNU/9. semester/Project thesis/github/Project-thesis/CHB_MIT" 

# signal processing
downsampling = 2 
Fs = 256//downsampling 
mother_wavelet = 'bior2.2'
dwt_levels = 4 # TODO: change name
dwt_decomp = True # choose dwt or emd decomposition 
emd_decomp = not(dwt_decomp)
segment_length = 6
unwanted_channels = ["ECG", "VNS", "EKG1", "LUE", "EKG2", "LOC"] # TODO: forklaring på hvorfor
#wanted_channels = ["FC1", "FC2", "FC5", "FC6", "FT10", "CP1", "CP5", "P8", "PZ", "T7", "FZ", "CZ", "FP1", "F4", "P7"]

# machine learning
k_folds = 10 
learning_rate = 0.75

start_patient = 1
end_patient = 24

