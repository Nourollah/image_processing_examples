import cv2
import matplotlib.pyplot as plt
plt.gray()


# Read image file in default mode
Img = cv2.imread("hafez.jpeg", cv2.IMREAD_UNCHANGED)
# Read image file in gray scale mode
ImgGray = cv2.imread("hafez.jpeg", cv2.IMREAD_GRAYSCALE)
# Set threshold for map gray scale image to binary(black/white)
_, Binary = cv2.threshold(ImgGray, 128, 255, cv2.THRESH_BINARY)
# Plot image
plt.figure(1)
plt.subplot(1, 3, 1)
plt.imshow(Img)
plt.title("Original Image")
plt.subplot(1, 3, 2)
plt.imshow(ImgGray)
plt.title("Grayscale Image")
plt.subplot(1, 3, 3)
plt.imshow(Binary)
plt.title("Binary Image")
plt.show()

