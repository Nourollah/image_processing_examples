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
plt.subplot(131)
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
imgnF2[118:122, :] = 0
imgnF2[138:142, :] = 0
imgnF2[:, 108:112] = 0
imgnF2[:, 146:150] = 0
NoisyImgFd = np.abs(imgnF2)
Max = np.max(np.max(imgnF2))
IMGNotchLog = np.log(1 + NoisyImgFd)
MaxNotch = np.max(np.max(IMGNotchLog))
plt.subplot(132)
plt.imshow(IMGNotchLog / MaxNotch, cmap='gray', vmin=0, vmax=1)
plt.title("Notch mask")
# inverse fft
notchImgInv = np.abs(np.fft.ifft2(imgnF2))
plt.subplot(133)
plt.imshow(notchImgInv / np.max(np.max(notchImgInv)), cmap='gray', vmin=0, vmax=1)
plt.title("Reducing noise")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
plt.show()
