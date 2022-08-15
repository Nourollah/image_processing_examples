import cv2
from matplotlib import pyplot as plt

img = cv2.cvtColor(cv2.imread("Lenna.jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
imgCMY = 255 - img
plt.subplot(221)
plt.imshow(img)
plt.title("Colorful image")
plt.subplot(222)
plt.imshow(imgCMY[:, :, 0], cmap='gray')
plt.title("Cyan Channel")
plt.subplot(223)
plt.imshow(imgCMY[:, :, 1], cmap='gray')
plt.title("Magenta Channel")
plt.subplot(224)
plt.imshow(imgCMY[:, :, 2], cmap='gray')
plt.title("Yellow Channel")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
plt.show()
