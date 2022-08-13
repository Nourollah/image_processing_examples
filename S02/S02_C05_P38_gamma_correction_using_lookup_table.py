import cv2
import numpy as np


def imadjust(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invert_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invert_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# Read image file in gray scale mode
# imgOrg = cv2.imread("Confidential.jpg", cv2.IMREAD_GRAYSCALE)
img = np.array(cv2.imread("Confidential.jpg", cv2.IMREAD_GRAYSCALE))
ImEnh = imadjust(img, 1.8)
# img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
# Plot images
cv2.imshow("Orginal image", img)
cv2.imshow("Enhanced image", ImEnh)
cv2.waitKey(0)
cv2.destroyAllWindows()
