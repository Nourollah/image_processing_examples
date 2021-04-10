import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.gray()

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def gaussian(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = np.exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
    return base



s1 = 256
s2 = 256
v = 50
H = gaussian(v, (s1, s2))
# Read image from file
img = cv2.imread("Einstein.png", cv2.IMREAD_GRAYSCALE)
# Take Fourier transform
af = np.fft.fftshift(np.fft.fft2(img))
# Conv filter and mask in frequency mode
t = af * H
# Take real part of complex number
M_Id = np.abs(t)
# Take log from image
M_IdLog = np.log(1 + M_Id)
# Calc max of 2D image
Max = np.max(np.max(M_IdLog))
# Plot image
plt.subplot(121)
plt.imshow(M_IdLog / Max)
# Return from frequency to place
cfli1 = np.fft.ifft2(np.fft.ifftshift(t))
# Take real part of complex number
IFFTt = np.abs(cfli1)
# Calc max of 2D image
Max2 = np.max(np.max(IFFTt))
# Plot image
plt.subplot(122)
plt.imshow(IFFTt / Max2)
plt.show()
