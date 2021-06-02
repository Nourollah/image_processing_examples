import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read image from file in gray scale mode and  convert to numpy array
img = cv2.normalize(cv2.imread("11.png", cv2.IMREAD_GRAYSCALE).astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
gx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
Imggx = cv2.filter2D(img, -1, gx)
Imggy = cv2.filter2D(img, -1, gx.T)

M = np.sqrt(np.power(Imggx, 2) + np.power(Imggy, 2))
Imggy = cv2.normalize(M.astype("uint8"), None, 0, 256, cv2.NORM_MINMAX)

plt.imshow(M, cmap='gray')
plt.title("Both side sobel")
plt.show()
