
DATASET_PATH = "CHB_MIT" #"C:/Users/marie/OneDrive - NTNU/NTNU/9. semester/Project thesis/github/Project-thesis/CHB_MIT" 

START_PATIENT = 12
END_PATIENT = 13

# --- Signal processing  --- #
DOWNSAMPLING_RATE = 2
FS = 256//DOWNSAMPLING_RATE
MOTHER_WAVELET = 'bior2.2'
DWT_LEVELS = 4 
DWT_DECOMP = True # choose dwt or emd decomposition 
EMD_DECOMP = not(DWT_DECOMP)
SEGMENT_LENGTH = 6
UNWANTED_CHANNELS = ["ECG", "VNS", "EKG1", "LUE", "EKG2", "LOC"] # these are not EEG signals
#wanted_channels = ["FC1", "FC2", "FC5", "FC6", "FT10", "CP1", "CP5", "P8", "PZ", "T7", "FZ", "CZ", "FP1", "F4", "P7"] # for debugging

# --- Machine learning --- #
MODEL = "rf" # choose between "rf", "gb" or "svm"
TREE=False
if MODEL == "rf" or MODEL =="gb":
    TREE=True
K_FOLDS = 10 
LEARNING_RATE = 0.75
SVM_KERNEL = 'linear'

# --- Files for saving results --- #
# Feature importance:
FILE_WITHOUT_NANS = 'features_no_nans.csv'
FEAT_FILE_FEATURES = "features_no_nans.csv"
FEATURE_NAMES = ["petrosian_fractal_dimension"]#, "std", "rms", "katz_fractal_dimension"]#var", "ptp_amp", "kurtosis", "instantaneous_energy", "teager_energy", "higuchi_fractal_dimension","petrosian_fractal_dimension", "std", "skewness", "line_length", "rms", "hjort_mobility", "hjort_complexitY", "katz_fractal_dim", "sevcik_fraction_dim","mean"] # names needed for plotting
IMPORTANCE_FILE_FEATURES = 'importance_features_sorted.txt'

# Channel importance:
FEAT_FILE_CHANNELS = "features_c.txt" 
LABELS_FILE = "channel_labels.txt"
CHANNELS_FILE = "channel_names_c.txt"
CHANNEL_NAMES = []
IMPORTANCE_FILE_CHANNELS = 'importance_channels_sorted.txt'

# Accuracy plotting
ACCURACY_FILE = "accuracy.txt"

# --- Plots --- #
# Performance
HEATMAP = "heatmap"
FEATURE_IMPORTANCE_HIST = "feature_importances_hist.pdf"
CHANNEL_IMPORTANCE_HIST = "channel_importances_hist.pdf"


