import cv2
from skimage import color as scl
from matplotlib import pyplot as plt


imgRGB = cv2.cvtColor(cv2.imread("Lenna.jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
imgHSV = scl.rgb2hsv(imgRGB)
imgYIQ = scl.rgb2yiq(imgRGB)

plt.subplot(221)
plt.imshow(imgRGB)
plt.title("Original Image")

imgRGB[:, :, 0] = cv2.equalizeHist(imgRGB[:, :, 0])
imgRGB[:, :, 1] = cv2.equalizeHist(imgRGB[:, :, 1])
imgRGB[:, :, 2] = cv2.equalizeHist(imgRGB[:, :, 2])
plt.subplot(222)
plt.imshow(imgRGB)
plt.title("RGB")



imgHSV_8uc1 = cv2.normalize(imgHSV, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
imgHSV_8uc1[:, :, 2] = cv2.equalizeHist(imgHSV_8uc1[:, :, 2])
plt.subplot(223)
plt.imshow(scl.hsv2rgb(imgHSV_8uc1))
plt.title("HSV")

imgYIQ_8uc1 = cv2.normalize(imgYIQ, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
imgYIQ_8uc1[:, :, 0] = cv2.equalizeHist(imgYIQ_8uc1[:, :, 0])
plt.subplot(224)
plt.imshow(scl.yiq2rgb(imgYIQ_8uc1))
plt.title("YIQ")


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
plt.show()
