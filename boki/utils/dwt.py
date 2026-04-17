import pywt

def dwt(data, wavelet, level):
  (cA, cD) = pywt.dwt(data, wavelet)
  for i in range(level-1):
    (cA, cD) = pywt.dwt(cA, wavelet)

  return cA