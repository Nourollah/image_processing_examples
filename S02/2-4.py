import cv2
import numpy as np

# Read image file in gray scale mode
# imgOrg = cv2.imread("Confidential.jpg", cv2.IMREAD_GRAYSCALE)
img = np.array(cv2.imread("Confidential.jpg", cv2.IMREAD_GRAYSCALE))
img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
# Define Gamma value
Gamma = 1.8
ImEnh = np.float_power(img, Gamma)
# cv2.imshow("Orginal Image", imgOrg)
cv2.imshow("Enhanced Image", ImEnh)
cv2.waitKey(0)
cv2.destroyAllWindows()

