import cv2
from scipy.signal.signaltools import wiener
from skimage.util import random_noise
from matplotlib import pyplot as plt

img = cv2.normalize(cv2.imread("Lenna.jpg", cv2.NORM_MINMAX).astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
imgn = random_noise(img)
imgd1 = wiener(imgn, (3, 3))
imgd2 = wiener(imgn, (5, 5))
imgd3 = wiener(imgn, (7, 7))

plt.subplot(131)
plt.imshow(imgd1)
plt.subplot(132)
plt.imshow(imgd2)
plt.subplot(133)
plt.imshow(imgd3)
plt.show()