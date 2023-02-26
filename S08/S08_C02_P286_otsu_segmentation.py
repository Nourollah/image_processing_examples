import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image
im = cv2.imread('hafez.jpeg', cv2.IMREAD_GRAYSCALE)

# Plot image and histogram
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(im, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.hist(im.ravel(), bins=256, range=(0, 255))
plt.title('Histogram')
plt.xlabel('Intensity Value')
plt.ylabel('Pixel Count')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()

# Calculate Otsu's threshold
P = np.zeros((256, 1))
y, _ = np.histogram(im, bins=256, range=(0, 255))
p = y / np.size(im)
P = np.cumsum(p)
M = np.zeros((256, 1))
M[0] = 0
for i in range(1, 256):
    M[i] = M[i - 1] + (i - 1) * p[i]

m_G = M[-1]
sigma2_B = np.zeros((256, 1))

for T in range(256):
    sigma2_B[T] = ((m_G * P[T] - M[T]) ** 2) / (P[T] * (1 - P[T]))

threshold = np.argmax(sigma2_B)
R = (im > threshold).astype(np.uint8) * 255

# Plot segmented image
plt.figure()
plt.imshow(R, cmap='gray')
plt.title('Otsu Thresholding')
plt.axis('off')
plt.show()
