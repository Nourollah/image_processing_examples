import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read image from file in gray scale mode and  convert to numpy array
img = cv2.normalize(cv2.imread("11.png", cv2.IMREAD_GRAYSCALE).astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
gx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
Imggx = cv2.filter2D(img, -1, gx)
Imggy = cv2.filter2D(img, -1, gx.T)

Imggx = cv2.normalize(Imggx.astype("uint8"), None, 0, 256, cv2.NORM_MINMAX)
Imggy = cv2.normalize(Imggy.astype("uint8"), None, 0, 256, cv2.NORM_MINMAX)

plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.subplot(132)
plt.imshow(Imggx, cmap='gray')
plt.title("Sobel in X")
plt.subplot(133)
plt.imshow(Imggy, cmap='gray')
plt.title("Sobel in Y")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
