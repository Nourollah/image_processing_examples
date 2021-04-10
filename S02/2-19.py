import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read image from file in gray scale mode and  convert to numpy array
img = np.array(cv2.imread("4.jpg", cv2.IMREAD_GRAYSCALE))
img1 = cv2.medianBlur(img, 3)
img2 = cv2.medianBlur(img, 7)


plt.subplot(121)
plt.imshow(img1, cmap='gray')
plt.title("Median filter 3*3")
plt.subplot(122)
plt.imshow(img2, cmap='gray')
plt.title("Median filter 7*7")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
