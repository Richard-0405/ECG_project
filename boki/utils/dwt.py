import numpy as np
import pywt


def dwt(signal, wavelet="db1", level=1):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.asarray(coeffs[0], dtype=float)
