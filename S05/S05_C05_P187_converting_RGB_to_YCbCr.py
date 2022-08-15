import cv2
from skimage import color as scl
from matplotlib import pyplot as plt

img = cv2.cvtColor(cv2.imread("Lenna.jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
imgYCBCR = scl.rgb2ycbcr(img)

plt.subplot(221)
plt.imshow(img)
plt.title("Colorful image")
plt.subplot(222)
plt.imshow(imgYCBCR[:, :, 0], cmap='gray')
plt.title("Y")
plt.subplot(223)
plt.imshow(imgYCBCR[:, :, 1], cmap='gray')
plt.title("CB")
plt.subplot(224)
plt.imshow(imgYCBCR[:, :, 2], cmap='gray')
plt.title("CR")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
plt.show()
