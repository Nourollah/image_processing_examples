# Install in (ipython, jupyter notebook, jupyter lab) ! pip3 install PyWavelets
# Install in conda ! conda install PyWavelets
from scipy.io import wavfile
import pywt

wn = 'sym4'
[fs, S] = wavfile.read("Sound.wav")
[CA, CD] = pywt.dwt(S, wn)
Res = pywt.idwt(CA, CD, wn)

