import config
from feature_calculation import calculate_features 
from signal_decomposition import emd_decomposition, dwt_decomposition

def get_features_from_signal(signal, feature_vector, seizure):
    '''
    Get features from signal interval.
    Do DWT on the signal and calculate features on the levels.
    
        param signal: signal to do find features from  
        param fature_vector: the vector to add more features to, list
        param seizure: wether it is a seizure (1) or not (0), 1
        
        return feature_vector: updated feature_vector with more features, list (of lists if it is a seizure)
    '''
    if config.DWT_DECOMP: 
        decomposition = dwt_decomposition(signal)
    elif config.EMD_DECOMP: 
        decomposition = emd_decomposition(signal)
    feature_vector = calculate_features(decomposition, feature_vector, seizure) 
    return feature_vector