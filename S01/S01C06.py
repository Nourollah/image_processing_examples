import cv2

# Read image from file in default mode
im = cv2.imread("veresk-bridge.jpg", cv2.IMREAD_GRAYSCALE)
# Find image shape, stored in dim variable
dim = im.shape
# dim1 used for resize image to half
dim1 = (int(dim[0] / 2), int(dim[1] / 2))
im1 = cv2.resize(im, dim1, interpolation=cv2.INTER_LINEAR)
# dim2 used for resize image to twice
dim2 = (int(dim1[0] * 2), int(dim1[1] * 2))
im2 = cv2.resize(im1, dim2, interpolation=cv2.INTER_LINEAR)
# Plot images
cv2.imshow("Original Image", im)
cv2.imshow("0.5x resized Image", im1)
cv2.imshow("2x resized Image", im2)
# Keep open images until press 0 key
cv2.waitKey(0)
cv2.destroyAllWindows()

