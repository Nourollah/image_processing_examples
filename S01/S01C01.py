import cv2

# Read image file in default mode
Img = cv2.imread("hafez.jpeg")
# Read image file in gray scale mode
ImgGray = cv2.imread("hafez.jpeg", cv2.IMREAD_GRAYSCALE)
# Set threshold for map gray scale image to binary(black/white)
Binary = cv2.threshold(ImgGray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# Plot image
cv2.imshow("Original Image", Img)
cv2.imshow("Gray Image", ImgGray)
cv2.imshow("Binary Image", Binary)
# Keep open images until press 0 key
cv2.waitKey(0)
cv2.destroyAllWindows()

