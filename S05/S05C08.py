import cv2
from skimage import color as scl
from matplotlib import pyplot as plt


canny_min=0
canny_max=90
img=cv2.imread("Lenna.jpg", cv2.IMREAD_COLOR)
imgGray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgHSV = scl.rgb2hsv(imgRGB)
imgYIQ = scl.rgb2yiq(imgRGB)


edgeGray=cv2.Canny(imgGray,canny_min, canny_max)

plt.subplot(221)
plt.imshow(edgeGray)
plt.title("Gray")

edR = cv2.Canny(imgRGB[:, :, 0],canny_min, canny_max)
edG = cv2.Canny(imgRGB[:, :, 1],canny_min, canny_max)
edB = cv2.Canny(imgRGB[:, :, 2],canny_min, canny_max)
edRGB=edR+edG+edB

plt.subplot(222)
plt.imshow(edRGB)
plt.title("RGB")



imgHSV_8u = cv2.normalize(imgHSV, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

edH = cv2.Canny(imgHSV_8u[:, :, 0],canny_min, canny_max)
edS = cv2.Canny(imgHSV_8u[:, :, 1],canny_min, canny_max)
edV = cv2.Canny(imgHSV_8u[:, :, 2],canny_min, canny_max)
edHSV=edS+edV
plt.subplot(223)
plt.imshow(edHSV)
plt.title("HSV")


imgYIQ_8u = cv2.normalize(imgYIQ, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

edY = cv2.Canny(imgYIQ_8u[:, :, 0],canny_min, canny_max)
edI = cv2.Canny(imgYIQ_8u[:, :, 1],canny_min, canny_max)
edQ = cv2.Canny(imgYIQ_8u[:, :, 2],canny_min, canny_max)
edYIQ=edY+edI+edQ

plt.subplot(224)
plt.imshow(edYIQ)
plt.title("YIQ")

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
plt.show()