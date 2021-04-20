import cv2
import skimage
import scipy.ndimage
import matplotlib.pyplot as plt

plt.gray()

img = cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE)
imgn = skimage.util.random_noise(skimage.util.random_noise(img, 'salt'), 'pepper')
imgd1 = scipy.ndimage.rank_filter(imgn, rank=8, size=3)
imgd2 = scipy.ndimage.rank_filter(imgn, rank=0, size=3)

plt.subplot(131)
plt.imshow(imgn)
plt.subplot(132)
plt.imshow(imgd1)
plt.subplot(133)
plt.imshow(imgd2)
plt.show()