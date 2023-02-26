import cv2
import numpy as np
from matplotlib import pyplot as plt

im = cv2.imread('AzadiTowerGray.jpg')

imcolor = im.copy()

if len(im.shape) == 3:
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

im = np.float32(im)

# Define a 3x3 averaging filter kernel
h1 = np.ones((3, 3), np.float32) / 9

# Apply the filter to the image with border replication
imd = cv2.filter2D(im, -1, h1, borderType=cv2.BORDER_REPLICATE)

# Get the row and column size of the image
row, col = im.shape

# Reshape the filtered image into a column vector
imV = imd.reshape(row * col, 1)

# Specify the number of clusters for k-means clustering
Num = 3

# Define the termination criteria for k-means clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Define the flags for k-means clustering
flags = cv2.KMEANS_RANDOM_CENTERS

# Perform k-means clustering on the reshaped image vector
compactness, labels, centers = cv2.kmeans(imV, Num, None, criteria, 10, flags)

# Reshape the cluster labels into an image
R = np.uint8(labels.reshape(row, col))

# Display the original and clustered images side by side
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(imcolor, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2), plt.imshow(R, cmap='gray')
plt.show()
