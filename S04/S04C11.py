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
# Apply Fourier transform to image with one noise
imgnF2 = np.fft.fftshift(np.fft.fft2(imgn1))
imgnFD2 = np.abs(imgnF2)
imgnFDL2 = np.log(imgnFD2)
Max2 = np.max(np.max(imgnFDL2))
imgSF2 = imgnFDL2 / Max2
# Remove noise in fourier series with mask
[x, y] = np.meshgrid(np.arange((-n / 2), (n / 2)), np.arange((-m / 2), (m / 2)))
r = np.sqrt(x ** 2 + y ** 2).reshape(1, -1)
z = np.zeros([256, 256]).reshape(1, -1)
mask = ((r[:, :] < 20) | (r[:, :] > 24))
z[np.where(mask)] = 1
filterimage = imgnF2.reshape(1, -1) * z
IMgFd8 = np.abs(filterimage.reshape(256, 256))
IMGLog8 = np.log(1 + IMgFd8)
Max8 = np.max(np.max(IMGLog8))
plt.subplot(132)
plt.imshow(IMGLog8 / Max8, cmap='gray', vmin=0, vmax=1)
plt.title("Mask for filtering")
# Plot filtered image with mask
imginv = np.abs(np.fft.ifft2(filterimage.reshape(256, 256)))
plt.subplot(133)
plt.imshow(imginv / np.max(np.max(imginv)), cmap='gray')
plt.title("Image with reducing noise")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()