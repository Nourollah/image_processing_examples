import cv2
import numpy as np
from matplotlib import pyplot as plt

im = cv2.imread('AzadiTowerGray.jpg')

# Creating a copy of the original image
imcolor = im.copy()

# Converting the image to grayscale if it is in color
if len(im.shape) == 3:
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Converting the image to a float32 data type
im = np.float32(im)

# Creating a 3x3 matrix with all elements being 1/9
h1 = np.ones((3, 3), np.float32)/9

# Filtering the image using the above matrix with border replication
imd = cv2.filter2D(im, -1, h1, borderType=cv2.BORDER_REPLICATE)

# Extracting the number of rows and columns of the image
row, col = im.shape

# Reshaping the filtered image to a 1D array
imV = imd.reshape(row*col, 1)

# Applying k-means clustering with k=3
Num = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(imV, Num, None, criteria, 10, flags)

# Reshaping the labels array to a 2D array with dimensions same as the original image
R = np.uint8(labels.reshape(row, col))

# Displaying the original and clustered images using matplotlib
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(imcolor, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2), plt.imshow(R, cmap='gray')
plt.show()
