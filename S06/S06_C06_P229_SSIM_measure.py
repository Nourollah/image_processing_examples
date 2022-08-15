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
# Local win for statistics
win = np.ones(8)
# Constant in the SSIM index formula
C1 = 25
C2 = 25
C3 = 25
win = win / np.sum(win[:])
# Average of image 1
M1 = cv2.filter2D(im1, -1, win, borderType=cv2.BORDER_REPLICATE)
M1p2 = np.multiply(M1, M1)
# Average of image 2
M2 = cv2.filter2D(im2, -1, win, borderType=cv2.BORDER_REPLICATE)
M2p2 = np.multiply(M2, M2)

M1_D_M2 = np.multiply(M1, M2)
# Variance of image 1
Var1p2 = cv2.filter2D(np.multiply(im1, im1), -1, win, borderType=cv2.BORDER_REPLICATE) - M1p2
# Variance of image 2
Var2p2 = cv2.filter2D(np.multiply(im2, im2), -1, win, borderType=cv2.BORDER_REPLICATE) - M2p2
# Covariance of image 2
Var12p2 = cv2.filter2D(np.multiply(im1, im2), -1, win, borderType=cv2.BORDER_REPLICATE) - M1_D_M2
SSIMMap = ((2 * M1_D_M2 + C1) / (M1p2 + M2p2 + C1)) * \
          ((2 * np.sqrt(Var1p2) * np.sqrt(Var2p2) + C2) / (Var1p2 + Var2p2 + C2)) * \
          ((Var12p2 + C3) / (np.sqrt(Var1p2) * np.sqrt(Var2p2) + C3))
msssim = np.mean(np.mean(SSIMMap))
# Plot image
plt.figure(2)
plt.imshow(SSIMMap)

