import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image from file in gray scale mode and  convert to numpy array
img = np.array(cv2.imread("4.jpg", cv2.IMREAD_GRAYSCALE))
# Plot histograms
plt.subplot(311)
plt.hist(img.ravel(), bins=256, range=[0, 256])
plt.subplot(312)
plt.hist(img.ravel(), bins=50, range=[0, 256])
img = np.histogram(img, range(0, 256)) / np.size(img)
plt.subplot(313)
plt.hist(img.ravel(), bins=50, range=[0, 256])
plt.xlim([0, 256])
plt.ylim([0, 1])
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()


