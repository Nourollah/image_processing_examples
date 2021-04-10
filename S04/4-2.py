import cv2
import skimage  # Used skimage as noise functions
import matplotlib.pyplot as plt

# Read image anc normalized it between 0.0 and 1.0 as float type
img = cv2.normalize(cv2.imread("Lenna.jpg").astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
# Apply gaussian noise to image
# mean and var parameter used in 'gaussian' and 'speckle' noise
imgn = skimage.util.random_noise(img, 'gaussian', mean=0, var=0.01)
# Plot images
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("Image without noise")
plt.subplot(122)
plt.imshow(imgn, cmap='gray')
plt.title("Image with noise")
plt.show()
