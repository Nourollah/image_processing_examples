import cv2
import numpy as np

# Read Original image --> this image doesn't have noise
im1 = cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE)
# Read Noisy image
im2 = cv2.imread("Noisy.jpg", cv2.IMREAD_GRAYSCALE)
# Calc Worst Case Absolute Error
WCAE = np.max(np.max(np.abs(im2 - im1)))

