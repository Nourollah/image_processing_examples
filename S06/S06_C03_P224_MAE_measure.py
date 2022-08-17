import cv2
import numpy as np

# Read Original image --> this image doesn't have noise
im1 = cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE)
# Read Noisy image
im2 = cv2.imread("Noisy.jpg", cv2.IMREAD_GRAYSCALE)
# Find image shape
[M, N] = im1.shape
# Calc Mean Absolute Error
MAE = np.sum(np.sum(np.abs(im2 - im1))) / M * N

