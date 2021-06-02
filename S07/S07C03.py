# Install in (ipython, jupyter notebook, jupyter lab) ! pip3 install PyWavelets
# Install in conda ! conda install PyWavelets
import pywt
from matplotlib import pyplot as plt
import cv2
plt.gray()


# Read Image
HostImage = cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE)
# Read Watermark
Watermark = cv2.resize(cv2.imread("Logo.jpg", cv2.IMREAD_GRAYSCALE), HostImage.shape)
wn = 'sym4'
# Discrete Wavelet Transform on image
coeffsImg = pywt.dwt2(HostImage, wn)
CA, (CV, CH, CD) = coeffsImg
# Discrete Wavelet Transform on Watermark
coeffsWM = pywt.dwt2(Watermark, wn)
CAW1, (CVW1, CHW1, CDW1) = coeffsWM
# Put watermark inside of image
coeffsWMI = CA, (CVW1, CHW1, CDW1)
# Create watermarked image
WaterMarkedImage = pywt.idwt2(coeffsWMI, wn)
# Extract watermark image
coeffsWMIE = pywt.dwt2(WaterMarkedImage, wn)
CAW2, (CVW2, CHW2, CDW2) = coeffsWMIE
CHW2[:, :] = 0
coeffsWMIE = CAW2, (CVW2, CHW2, CDW2)
ExtractedWaterMark = pywt.idwt2(coeffsWMIE, wn)
plt.subplot(221)
plt.imshow(HostImage)
plt.title("Host Image")
plt.subplot(222)
plt.imshow(Watermark)
plt.title("WaterMark Image")
plt.subplot(223)
plt.imshow(WaterMarkedImage.astype("uint8"))
plt.title("WaterMarked Image")
plt.subplot(224)
plt.imshow(ExtractedWaterMark)
plt.title("Image without watermark")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
plt.show()


