import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image from file in gray scale mode and  convert to numpy array
img = np.array(cv2.imread("11.png", cv2.IMREAD_GRAYSCALE))
# Normalize image with histogram equalization
img2 = cv2.equalizeHist(img)
# Plot images and histogram
plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title("Original image")
plt.subplot(223)
plt.hist(img.ravel(), bins=256, range=[0, 256])
plt.title("Original image histogram")
plt.subplot(222)
plt.imshow(img2, cmap='gray')
plt.title("Equalization image")
plt.subplot(224)
plt.hist(img2.ravel(), bins=256, range=[0, 256])
plt.title("Equalization image histogram")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
