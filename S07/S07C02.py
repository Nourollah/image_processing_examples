# Install in (ipython, jupyter notebook, jupyter lab) ! pip3 install PyWavelets
# Install in conda ! conda install PyWavelets
import pywt
from matplotlib import pyplot as plt
import cv2

img = cv2.imread("Lenna.jpg")
wn = 'sym4'
coeffs = pywt.dwt2(img, wn)
CA, (CV, CH, CD) = coeffs
ReImg = pywt.idwt2(coeffs, wn)
plt.imshow(ReImg.astype("uint8"))
plt.show()


