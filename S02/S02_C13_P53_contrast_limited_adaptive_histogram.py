import cv2
import matplotlib.pyplot as plt

# Read image from file in gray scale mode and  convert to numpy array
img = cv2.imread("6.jpg", cv2.IMREAD_GRAYSCALE)
# Create a CLAHE object (Arguments are optional)
clahe = cv2.createCLAHE(2.0, (8, 8))
# Use CLAHE to Equalization image
img2 = clahe.apply(img)
# Plot images
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("Original image")
plt.subplot(122)
plt.imshow(img2, cmap='gray')
plt.title("Equalization image")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
