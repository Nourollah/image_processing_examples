import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.filters as skf

# Read image from file in gray scale mode and  convert to numpy array
img = cv2.imread("11.png", cv2.IMREAD_GRAYSCALE)
alpha = 0.2
kernel = np.array([[-alpha, alpha - 1, -alpha],
                  [alpha - 1, alpha + 5, alpha - 1],
                  [-alpha, alpha - 1, -alpha]])
Ip = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
Ip = cv2.normalize(Ip.astype("uint8"), None, 0, 255, cv2.NORM_MINMAX)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("Original image")
plt.subplot(122)
plt.imshow(Ip, cmap='gray')
plt.title("Image with not sharped edge")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
