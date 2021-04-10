import cv2
import skimage
import matplotlib.pyplot as plt

# Read image anc normalized it between 0.0 and 1.0 as float type
img = cv2.normalize(cv2.imread("Lenna.jpg").astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
# Apply salt and pepper noise to image
# amount parameter used in 'salt' and 'pepper' noise
imgn = skimage.util.random_noise(skimage.util.random_noise(img, 'salt', amount=0.05), 'pepper', amount=0.05)
# Plot images
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("Image without noise")
plt.subplot(122)
plt.imshow(imgn, cmap='gray')
plt.title("Image with noise")
plt.show()

