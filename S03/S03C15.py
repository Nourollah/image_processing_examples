import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.gray()

# Create an array for using to make Butterworth high pass filter
[x, y] = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
# Set radius and filter times
R = 30
n = 2
# Apply Butterworth high pass filter formula to H array
H1 = 1 / (1 + ((np.sqrt(x ** 2 + y ** 2) / R) ** (2 * n)))
H = 1 - H1
# Read image from file
img = cv2.imread("Einstein.png", cv2.IMREAD_GRAYSCALE)
# Take Fourier transform
imgF = np.fft.fftshift(np.fft.fft2(img))
# Conv filter and mask in frequency mode
M_I = imgF * H
# Take real part of complex number
M_Id = np.abs(M_I)
# Take log from image
M_IdLog = np.log(1 + M_Id)
# Calc max of 2D image
Max = np.max(np.max(M_IdLog))
# Plot image
plt.subplot(121)
plt.imshow(M_IdLog / Max)
# Return from frequency to place
cfli1 = np.fft.ifft2(np.fft.ifftshift(M_I))
# Take real part of complex number
IFFTt = np.abs(cfli1)
# Calc max of 2D image
Max2 = np.max(np.max(IFFTt))
# Plot image
plt.subplot(122)
plt.imshow(IFFTt / Max2)
plt.show()
