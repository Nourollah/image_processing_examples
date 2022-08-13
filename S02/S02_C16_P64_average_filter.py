import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image from file in gray scale mode and  convert to numpy array
img = np.array(cv2.imread("11.png", cv2.IMREAD_GRAYSCALE))
imgd = cv2.normalize(img.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
img1 = cv2.blur(imgd, ksize=(3, 3), borderType=cv2.BORDER_REPLICATE)
img2 = cv2.blur(imgd, ksize=(7, 7), borderType=cv2.BORDER_REPLICATE)
img3 = cv2.blur(imgd, ksize=(21, 21), borderType=cv2.BORDER_REPLICATE)

plt.subplot(221)
plt.imshow(imgd, cmap='gray')
plt.title("Original image")
plt.subplot(222)
plt.imshow(img1, cmap='gray')
plt.title("3*3 Filter")
plt.subplot(223)
plt.imshow(img2, cmap='gray')
plt.title("7*7 Filter")
plt.subplot(224)
plt.imshow(img3, cmap='gray')
plt.title("21*21 Filter")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
