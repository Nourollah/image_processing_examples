import cv2
import numpy as np

# Read image from file in default mode
img = cv2.imread('AzadiTower.jpg', 0)
# Make empty list to buffering bit plane
out = []
# Use loop to Select related bits to put in their own bit plane
for k in range(0, 8):
    # create an image for the k bit plane
    plane = np.full((img.shape[0], img.shape[1]), 2 ** k, dtype=np.uint8)
    # execute bitwise and operation
    res = cv2.bitwise_and(plane, img)
    # multiply ones (bit plane sliced) with 255 just for better visualization
    x = res * 255
    # append to the output list
    out.append(x)

# Show every bit plane as separated images
for i in range(0, 8):
    # Convert [0 - 255] to [0.0 - 1.0]
    out[i] = cv2.normalize(out[i].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow(f"bit plane {i}", np.array(out[i]))

# Recovery original image from bit planes
OriginalImage = 2 * (2 * (2 * (
        2 * (2 * (2 * (2 * np.array(out[7]) + np.array(out[6])) + np.array(out[5])) + np.array(out[4])) +
        np.array(out[3])) + np.array(out[2])) + np.array(out[1])) + np.array(out[0])
# Convert [0 - 255] to [0.0 - 1.0]
OriginalImage = cv2.normalize(OriginalImage.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
# Plot image
cv2.imshow("Original image", OriginalImage)
# Keep open images until press 0 key
cv2.waitKey(0)
cv2.destroyAllWindows()
