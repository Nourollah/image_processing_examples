import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.gray()

# Original Image
plt.figure(1)
img = cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE)
imgD = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
# Fourier transform image
imgF1 = np.fft.fftshift(np.fft.fft2(imgD))
imgFd1 = np.abs(imgF1)
imgFdL1 = np.log(imgFd1)
Max1 = np.max(np.max(imgFdL1))
# Apply periodic noise to image
plt.subplot(221)
[m, n] = imgD.shape
[a, b] = np.meshgrid(np.arange(0, n), np.arange(0, m))
p = np.sin(a / 2 + b / 4) + 1
imgn1 = cv2.normalize((imgD + p) / 2, None, 0.0, 1.0, cv2.NORM_MINMAX)
plt.title("Noisy Image")
plt.imshow(imgn1)
# Apply Fourier transform to noisy image
imgnF2 = np.fft.fftshift(np.fft.fft2(imgn1))
imgnFD2 = np.abs(imgnF2)
imgnFDL2 = np.log(imgnFD2)
Max2 = np.max(np.max(imgnFDL2))
imgSF2 = imgnFDL2 / Max2
# Remove noise in fourier series with notch filter
w = 15
T = 0.01
theta = np.angle(imgnF2)
IMgFd = np.abs(imgnF2)
IMGLog = np.log(1 + IMgFd)
Max = np.max(np.max(IMGLog))
plt.subplot(222)
plt.imshow(IMGLog / Max)
Fimage = IMgFd
for kk in range(np.floor(w / 2 + 1).astype("int"), (m - np.floor(w / 2)).astype("int")):
    for j in range(np.floor(w / 2 + 1).astype("int"), (n - np.floor(w / 2)).astype("int")):
        win = IMgFd[(kk - np.floor(w / 2)):(kk + np.ceil(w / 2 - 1)), (j - np.floor(w / 2)): (j + np.ceil(w / 2 - 1))]
        m = np.median(win[:])
        # o = np.zeros([kk, j])
        # o =
        if (IMgFd[kk, j] / m) >= T:
            Fimage[kk, j] = m
        else:
            Fimage[kk, j] = IMgFd[kk, j]

Z = Fimage * np.exp()
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
plt.show()
