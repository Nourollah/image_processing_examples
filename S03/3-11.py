import cv2
import numpy as np
import matplotlib.pyplot as plt
# Get to gray color map for plots
plt.gray()

# Create an array for using to make Ideal high pass filter
[x, y] = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
# Take square root from sum of two x and y and make new array with that
D = np.sqrt(x ** 2 + y ** 2)
# save filter 2D size
HShape = D.shape
# Reshape to (1, n) for working easy with that
D = D.reshape(1, -1)
# Create filter and init with zeros
H = np.zeros(shape=HShape).reshape(1, -1)
# Find correct range for put variables into filter
x = (D[:, :] > 30).reshape(1, -1)
# reinitialize filter to make correct high pass filter
H[np.where(x)] = D[x]
# Convert to 2D filter
H = H.reshape(HShape)
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
IFFT = np.fft.ifft2(np.fft.ifftshift(M_I))
# Take real part of complex number
IFFTt = np.abs(IFFT)
# Calc max of 2D image
Max2 = np.max(np.max(IFFTt))
# Plot image
plt.subplot(122)
plt.imshow(IFFTt / Max2)
plt.show()
