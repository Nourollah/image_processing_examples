import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.gray()

# Read Original image --> this image doesn't have noise
im1 = cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE)
# Convert to Double
im1 = cv2.normalize(im1.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
# Find Shape of image 1 (Noiseless image)
[m, n] = im1.shape
# Read Noisy Image
im2 = cv2.imread("Noisy.jpg", cv2.IMREAD_GRAYSCALE)
# Resize image 2 to size image 1
im2 = cv2.resize(im2, [m, n])
# Convert to Double
im2 = cv2.normalize(im2.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
# Plot images
plt.figure(1)
plt.subplot(121)
plt.imshow(im1)
plt.subplot(122)
plt.imshow(im2)
plt.show()
# w is the size of window
w = 2
NVF = np.ones(im1.size).reshape(im1.shape)

for i in range(np.ceil(w / 2).astype(int), (np.size(im1, 0) - np.floor(w / 2)).astype(int)):
    for j in range(np.ceil(w / 2).astype(int), (np.size(im1, 1) - np.floor(w / 2)).astype(int)):
        win = im1[(i - np.ceil(w / 2) + 1).astype(int):(i + np.floor(w / 2)).astype(int),
              (j - np.ceil(w / 2) + 1).astype(int):(j + np.floor(w / 2)).astype(int)]
        NVF[i, j] = 1 / (1 + np.var(win[:]))

WMSE = np.sum(np.sum(NVF * ((im1 - im2) ** 2))) / m * n
