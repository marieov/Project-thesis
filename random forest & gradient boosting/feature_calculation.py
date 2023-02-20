import mne_features.univariate as mne # https://mne.tools/mne-features/api.html
import hfda
import neurokit2 as nk
import numpy as np

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
    
def mean(data):
    mean = mne.compute_mean(data)
    return mean
    
    
def variance(data):
    return mne.compute_variance(data)
  
    
def std(data): 
    return mne.compute_std(data)


def skewness(data):
    return mne.compute_skewness(data)


def zero_crossing(data):
    return mne.compute_zero_crossings(data, threshold = 2.221e-16)


def kurtosis(data):
    return mne.compute_kurtosis(data)


def spect_entropy(data):
    return mne.compute_spect_entropy(250, data)


def higuchi_fractal_dimension(data, k_max=None):
    return hfda.measure(data, 10)
    
    
def petrosian_fractal_dimension(data):
    N_delta = np.diff(np.signbit(data)).sum()
    N = len(data)
    return (np.log10(N))/(np.log10(N) + np.log10((N/N+0.4*N_delta)))


def ptp_amp(data):
    return mne.compute_ptp_amp(data)


def line_length(data): 
    return mne.compute_line_length(data)


def rms(data): 
    return mne.compute_rms(data)


def hurst_exp(data): 
    return mne.compute_hurst_exp(data)


def quantile(data):
    return mne.compute_quantile(data)
    
    
def approximate_entropy(data):
    return mne.compute_app_entropy(data)


def sample_entropy(data):
    return mne.compute_samp_entropy(data)


def decorrelation_time(data): 
    return mne.compute_decorr_time(data)


def hjort_mobility(data):
    return mne.compute_hjorth_mobility(data)


def hjort_complexity(data): 
    return mne.compute_hjorth_complexity(data)


def svd_entropy(data):
    return mne.compute_svd_entropy(data)


def svd_fisher_info(data):
    return mne.compute_svd_fisher_info(data)


def band_energy(data):
    return mne.compute_energy_freq_bands(data)


def spectral_edge_freq(data):
    return mne.compute_spect_edge_freq(data)


def teager_kaiser_energy(data):
    return mne.compute_teager_kaiser_energy(data)


def katz_fractal_dimension(data): 
    return nk.fractal_katz(data)


def sevcik_fraction_dimentsion(data): 
    return nk.fractal_sevcik(data)


def calculate_features(dwt, feature_vector, seizure):
    '''
    Calculating features from the decomposition levels. 
    If it is a seizure, the features are added as lists, if not the features ar eadded directly.

        param dwt: the dwt levels, list
        param fature_vector: the vector to add more features to
        param seizure: seizure (1) or no seizure (0), int

        return feature_vector: updated feature_vector with one more row of features, list (of lists if it is a seizrue)
    '''
    feature = []
    # for each channel: 
    for dwt_value in dwt:
        katz_fractal_dim, _ = katz_fractal_dimension(dwt_value)
        sevcik_fraction_dim, _ = sevcik_fraction_dimentsion(dwt_value)
        feature += [petrosian_fractal_dimension(dwt_value)
                    #std(dwt_value),
                    #rms(dwt_value),
		            #variance(dwt_value),
                    #ptp_amp(dwt_value), 
                    #kurtosis(dwt_value),
                    #instantaneous_energy(dwt_value),
                    #teager_energy(dwt_value),                     
                    #higuchi_fractal_dimension(dwt_value), 
                    #skewness(dwt_value),              
                    #line_length(dwt_value), 
                    #hjort_mobility(dwt_value), 
                    #hjort_complexity(dwt_value),
                    #katz_fractal_dim, 
                    #sevcik_fraction_dim,
                    #mean(dwt_value)
                    ] 
    if not seizure:
        feature_vector += feature
    else: 
        feature_vector.append(feature)
    return feature_vector