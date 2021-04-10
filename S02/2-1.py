import numpy as np
import cv2

# Read image file
img = cv2.imread("images.jpg", cv2.IMREAD_GRAYSCALE)
# Make numpy array from image
q = np.array(img).astype(np.float)
# Define formula consts
S = 0.9
m = 60.0
# eps prevent error from divide on zero
eps = np.finfo(float).eps
# Define formula -> P = T(q) = 1 / (1 + (m / q)^S
p = 1.0 / (1.0 + (m / img + eps) ** S)
# plot images
cv2.imshow("Original Image", img)
cv2.imshow("Changed Image", p)
cv2.waitKey(0)
cv2.destroyAllWindows()
