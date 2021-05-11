import cv2
from skimage import color as scl
from matplotlib import pyplot as plt

img = cv2.cvtColor(cv2.imread("Lenna.jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
imgYIQ = scl.rgb2yiq(img)

plt.subplot(221)
plt.imshow(img)
plt.title("Colorful image")
plt.subplot(222)
plt.imshow(imgYIQ[:, :, 0], cmap='gray')
plt.title("Y")
plt.subplot(223)
plt.imshow(imgYIQ[:, :, 1], cmap='gray')
plt.title("I")
plt.subplot(224)
plt.imshow(imgYIQ[:, :, 2], cmap='gray')
plt.title("Q")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
plt.show()