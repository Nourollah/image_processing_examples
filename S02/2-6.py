import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image from file in gray scale mode and  convert to numpy array
img = np.array(cv2.imread("images.jpeg", cv2.IMREAD_GRAYSCALE))
# Normal [0 - 255] to [0.0 - 1.0]
img = cv2.normalize(img.astype('float32'), None, 0, 255, cv2.NORM_MINMAX)
# Fourier transform
f = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(f)
# Take log from image in fourier series mode
NewImg = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
# Plot images
plt.subplot(311)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])
plt.subplot(312)
plt.imshow(NewImg, cmap='gray')
plt.title('Fourier log transform')
plt.xticks([])
plt.yticks([])
plt.subplot(313)
plt.imshow(dft_shift[:, :, 1], cmap='gray')
plt.title('Fourier Transform image')
plt.xticks([])
plt.yticks([])
plt.show()
