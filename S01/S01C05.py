import cv2
import matplotlib.pyplot as plt
plt.gray()


# Read image from file in default mode
im = cv2.imread("veresk-bridge.jpg", cv2.IMREAD_GRAYSCALE)
# Find image shape, stored in dim variable
dim = im.shape
# dim1 used for resize image to half
dim1 = (int(dim[0] / 2), int(dim[1] / 2))
im1 = cv2.resize(im, (dim1[1], dim[0]), interpolation=cv2.INTER_NEAREST)
# dim2 used for resize image to twice
dim2 = (int(dim1[0] * 2), int(dim1[1] * 2))
im2 = cv2.resize(im1, (dim2[1], dim2[0]), interpolation=cv2.INTER_NEAREST)
# Plot images
plt.subplot(1, 3, 1)
plt.imshow(im)
plt.title("Original Image")
plt.subplot(1, 3, 2)
plt.imshow(im1)
plt.title("0.5x resized Image")
plt.subplot(1, 3, 3)
plt.imshow(im2)
plt.title("2x resized Image")
plt.show()
