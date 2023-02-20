import pywt
from pyhht.emd import EMD
import config
import numpy as np


#This part is not used in the project, working more on it in the masters
def emd_decomposition(signal):
    signal = np.array([signal])
    imf = EMD(signal)
    EMD_levels = imf.decompose() 
    emd = EMD_levels[0:config.dwt_levels+1] # this is wrong, has to be choosen more carefully
    return emd


def dwt_decomposition(signal):
    dwt = pywt.wavedec(signal, config.mother_wavelet, level=config.dwt_levels)
    return dwt