import cv2
import numpy as np
from skimage.feature import canny
import matplotlib.pyplot as plt

plt.gray()

# Read Original image --> this image doesn't have noise
im1 = cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE)
# Find Shape of image 1 (Noiseless image)
[m, n] = im1.shape
# Read Noisy Image
im2 = cv2.imread("Noisy.jpg", cv2.IMREAD_GRAYSCALE)
# Resize image 2 to size image 1
im2 = cv2.resize(im2, [m, n])
# Edge detector on images
RefIm = canny(im1, 0.1)
TestIm = canny(im2, 0.1)
# Number of edge pixels in the test image
C = np.sum(TestIm[:])
Landa = 1.9
# [co[:]]
