import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image from file in gray scale mode and  convert to numpy array
img = np.array(cv2.imread("6.jpg", cv2.IMREAD_GRAYSCALE))
# Calc image histogram with 256 bins
Nhist = cv2.calcHist([img], [0], None, [256], [0, 256]) / np.size(img)
# Calc cumulative sum of the elements
CDFhist = np.cumsum(Nhist)
# Make new buffer to make image
img2 = np.zeros(img.shape).astype("uint8")
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        img2[i, j] = (CDFhist[img[i, j]])

img2 = cv2.normalize(img2.astype('float'), None, 0, 255, cv2.NORM_MINMAX)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("Original image")
plt.subplot(122)
plt.imshow(img2.astype("uint8"), 'gray')
plt.title("Equalization image")
plt.show()
# np.size(cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX))

# img2 = cv2.equalizeHist(img)
# plt.subplot(221)
# plt.imshow(img, cmap='gray')
# plt.title("Original image")
# plt.subplot(223)
# plt.hist(img.ravel(), bins=256, range=[0, 256])
# plt.title("Original image histogram")
# plt.subplot(222)
# plt.imshow(img2, cmap='gray')
# plt.title("Equalization image")
# plt.subplot(224)
# plt.hist(img2.ravel(), bins=256, range=[0, 256])
# plt.title("Equalization image histogram")
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
# plt.show()
