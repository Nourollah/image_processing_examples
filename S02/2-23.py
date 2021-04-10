import cv2
import matplotlib.pyplot as plt
import skimage.filters as skf

# Read image from file in gray scale mode and  convert to numpy array
im = cv2.imread("11.png", cv2.IMREAD_GRAYSCALE)
img = cv2.normalize(im.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
Imgf = skf.laplace(img)

Imgf = cv2.normalize(Imgf.astype("uint8"), None, 0, 256, cv2.NORM_MINMAX)

Ip = im + (Imgf)


plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.title("Original image")
plt.subplot(132)
plt.imshow(Imgf, cmap='gray')
plt.title("Edge with laplacian")
plt.subplot(133)
plt.imshow(Ip, cmap='gray')
plt.title("Sharped image")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()

