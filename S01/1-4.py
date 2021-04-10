import cv2
import numpy as np

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

# Plot image
cv2.imshow("Original Image", im)
cv2.imshow("Mask 1", mask1)
cv2.imshow("And Result", cv2.bitwise_and(img, mask1))
cv2.imshow("Mask 2", mask2)
cv2.imshow("Or Result", cv2.bitwise_or(img, mask2))
# Keep open images until press 0 key
cv2.waitKey(0)
cv2.destroyAllWindows()

