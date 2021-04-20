import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image from file in gray scale mode and  convert to numpy array
img = np.array(cv2.imread("11.png", cv2.IMREAD_GRAYSCALE))
src = np.array(cv2.imread("22.png", cv2.IMREAD_GRAYSCALE))
img2 = cv2.equalizeHist(img, src)
plt.subplot(231)
plt.imshow(img, cmap='gray')
plt.subplot(234)
plt.hist(img.ravel(), bins=256, range=[0, 256])
plt.subplot(232)
plt.imshow(src, cmap='gray')
plt.subplot(235)
plt.hist(src.ravel(), bins=256, range=[0, 256])
plt.subplot(233)
plt.imshow(img2, cmap='gray')
plt.subplot(236)
plt.hist(img2.ravel(), bins=256, range=[0, 256])
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
