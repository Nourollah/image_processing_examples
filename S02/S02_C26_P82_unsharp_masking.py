import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.filters as skf

# Read image from file in gray scale mode and  convert to numpy array
im = cv2.imread("11.png", cv2.IMREAD_GRAYSCALE)
img = cv2.normalize(im.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)

Iavr = cv2.blur(img, ksize=(3, 3), borderType=cv2.BORDER_REPLICATE)
g = img - Iavr
k = 2
Ip = img + k * g

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("Original image")
plt.subplot(122)
plt.imshow(Ip, cmap='gray')
plt.title("Image with sharped edge")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
