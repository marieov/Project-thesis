import pywt
from pyhht.emd import EMD
import config
import numpy as np


def dwt_decomposition(signal):
    dwt = pywt.wavedec(signal, config.MOTHER_WAVELET, level=config.DWT_LEVELS)
    return dwt

    
#This part is not used in the project, working more on it in the masters
def emd_decomposition(signal):
    signal = np.array([signal])
    imf = EMD(signal)
    EMD_levels = imf.decompose() 
    emd = EMD_levels[0:config.DWT_LEVELS+1] # this is wrong, has to be choosen more carefully
    return emd

