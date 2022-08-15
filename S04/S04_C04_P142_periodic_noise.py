import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image anc normalized it between 0.0 and 1.0 as float type
img = cv2.normalize(cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE).astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
# Create periodic noise
[m, n] = img.shape
[a, b] = np.meshgrid(np.arange(0, n), np.arange(0, m))
p = np.sin(a / 2 + b / 4) + 1
# Adding made noise to the image
imgn = cv2.normalize((img + p) / 2, None, 0, 255, cv2.NORM_MINMAX)
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("Image without noise")
plt.subplot(122)
plt.imshow(imgn, cmap='gray')
plt.title("Image with noise")
plt.show()
