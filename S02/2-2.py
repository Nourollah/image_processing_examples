import cv2


# Complement image function
def invert_to_complement(image):
    image = (255 - image)
    return image


# Read image file
img = cv2.imread("eays.jpg", cv2.IMREAD_GRAYSCALE)
# Use opencv to complement
img2 = cv2.bitwise_not(img)
# Use custom function to complement
# img3 = invert_to_complement(img)
# Plot images
cv2.imshow("Original image", img)
cv2.imshow("Complement image", img2)
# cv2.imshow("Custom complement image", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
