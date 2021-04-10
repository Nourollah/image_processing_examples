import cv2
import numpy as np

# Read image file in gray scale mode
Image = cv2.imread("Ultrasound.jpg", cv2.IMREAD_GRAYSCALE)
# Convert cv2 array to numpy array
img1 = np.array(Image)
# Take copy from original image
img2 = img1.copy()
img3 = img1.copy()
# Apply filter to img2 & img3 with different range
img2[img2 > 128] = 255 - img2[img2 > 128]
img3[img3 < 128] = 255 - img3[img3 < 128]
# Plot image
cv2.imshow("Original Image", img1)
cv2.imshow("Complement Image with 5-2 B", cv2.bitwise_not(img1))
cv2.imshow("Complement Image with 2-6 A", img2)
cv2.imshow("Complement Image with 2-6 B", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

