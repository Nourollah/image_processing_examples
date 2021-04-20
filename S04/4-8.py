import cv2
import numpy as np
import skimage.util
import matplotlib.pyplot as plt

plt.gray()

# Open image in grayscale mode. this image when opened, have only one channel
img = cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE)

n = skimage.util.random_noise(img, 'gaussian', mean=0, var=0.01)
plt.subplot(121)
plt.imshow(n)

win = np.array([5, 5])
n = cv2.normalize(n.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
localMean = cv2.filter2D(n, -1, win) / (win[0] * win[1])

localVariance = cv2.filter2D(n ** 2, -1, np.ones(win.shape)) / (win[0] * win[1]) - localMean ** 2
noiseVariance = np.mean(np.mean(localVariance))

r = localMean + (np.maximum(0, localVariance - noiseVariance)) / np.maximum(localVariance, noiseVariance) * (n - localMean)

r = cv2.normalize(r.astype("uint8"), None, 0, 255, cv2.NORM_MINMAX)
plt.subplot(122)
plt.imshow(r)
plt.show()