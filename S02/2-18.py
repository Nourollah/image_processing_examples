import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read image from file in gray scale mode and  convert to numpy array
img = np.array(cv2.imread("11.png", cv2.IMREAD_GRAYSCALE))
imgd = cv2.normalize(img.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
img1 = cv2.GaussianBlur(imgd, ksize=(3, 3), sigmaX=0.5, borderType=cv2.BORDER_REPLICATE)
img2 = cv2.GaussianBlur(imgd, ksize=(3, 3), sigmaX=1.0, borderType=cv2.BORDER_REPLICATE)
img3 = cv2.GaussianBlur(imgd, ksize=(7, 7), sigmaX=2.0, borderType=cv2.BORDER_REPLICATE)
img4 = cv2.GaussianBlur(imgd, ksize=(7, 7), sigmaX=7.0, borderType=cv2.BORDER_REPLICATE)

plt.subplot(221)
plt.imshow(img1, cmap='gray')
plt.title("sigma = 0.5, 3*3")
plt.subplot(222)
plt.imshow(img2, cmap='gray')
plt.title("sigma = 1.0, 3*3")
plt.subplot(223)
plt.imshow(img3, cmap='gray')
plt.title("sigma = 2.0, 7*7")
plt.subplot(224)
plt.imshow(img4, cmap='gray')
plt.title("sigma = 7.0, 7*7")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
