import cv2
import numpy as np

# Read Original image --> this image doesn't have noise
im1 = cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE)
# Find image shape
[M, N] = im1.shape
# Read Noisy image
im2 = cv2.imread("Noisy.jpg", cv2.IMREAD_GRAYSCALE)
# Calc Mean Absolute Error
temp = cv2.bitwise_xor(im1, im2)
S = np.sum(np.sum(temp))
BCR = 1 - S / M * N

