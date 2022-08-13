import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.filters as skf

# Read image from file in gray scale mode and  convert to numpy array
im = cv2.imread("11.png", cv2.IMREAD_GRAYSCALE)
img = cv2.normalize(im.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
# Apply Gaussian Blur
blur = cv2.GaussianBlur(im, (3, 3), sigmaX=2)
img1 = cv2.Laplacian(blur, cv2.CV_64F)

img1 = cv2.normalize(img1.astype("uint8"), None, 0, 256, cv2.NORM_MINMAX)
X, ImgB = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY)

plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.title("Original image")
plt.subplot(132)
plt.imshow(img1, cmap='gray')
plt.title("Edge with log")
plt.subplot(133)
plt.imshow(ImgB, cmap='gray')
plt.title("Binary image")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
