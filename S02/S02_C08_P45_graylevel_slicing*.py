import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image from file in gray scale mode and  convert to numpy array
im = np.array(cv2.imread("tomur.jpg", cv2.IMREAD_GRAYSCALE))
# Plot original image
plt.subplot(121)
plt.imshow(im, cmap='gray')
# Normal [0 - 255] to [0.0 - 1.0]
# Reshape to (1, -1) for simply calculation
img = np.array(cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)).reshape(1, -1)
# Define x and y axis value
a = np.array([0.0, 0.3, 0.3, 0.5, 0.5, 1.0])
b = np.array([0.0, 0.3, 0.8, 0.8, 0.5, 1.0])
# Make new array for making image from original image
out = np.zeros(img.shape).reshape(1, -1)
# Save a size in N
N = len(a)
# Replace out array value with relative value
for i in range(N - 1):
    pix = ((a[i] <= img) & (img < a[i + 1])).reshape(1, -1)
    out[np.where(pix)] = np.multiply((img[np.where(pix)] - a[i]), (b[i + 1] - b[i])) / (a[i + 1] - a[i]) + b[i]
pix = (img == a[N - 1]).reshape(1, -1)
out[np.where(pix)] = b[N - 1]
# Plot images
plt.subplot(122)
plt.imshow(out.reshape(im.shape), cmap='gray')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
