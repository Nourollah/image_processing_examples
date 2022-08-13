import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.gray()


# Read image from file in gray scale mode
im = cv2.imread("Car.jpg", cv2.IMREAD_GRAYSCALE)
# Convert image integers to decimals
img = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# Create mask for filtering
mask1 = np.full((im.shape[0], im.shape[1]), 0.0, np.float)
for x in range(557, 613):
    for y in range(258, 526):
        mask1[x, y] = 1.0

# Create mask for filtering
mask2 = np.full((im.shape[0], im.shape[1]), 1.0, np.float)
for x in range(557, 613):
    for y in range(258, 526):
        mask2[x, y] = 0.0

plt.figure(1)
plt.subplot(2, 3, 1)
plt.imshow(im)
plt.title("Original Image")
plt.subplot(2, 3, 4)
plt.imshow(im)
plt.title("Original Image")
plt.subplot(2, 3, 2)
plt.imshow(mask1)
plt.title("Mask 1")
plt.subplot(2, 3, 3)
plt.imshow(cv2.bitwise_and(img, mask1))
plt.title("And Result")
plt.subplot(2, 3, 5)
plt.imshow(mask2)
plt.title("Mask 2")
plt.subplot(2, 3, 6)
plt.imshow(cv2.bitwise_or(img, mask2))
plt.title("Or Result")
plt.show()
