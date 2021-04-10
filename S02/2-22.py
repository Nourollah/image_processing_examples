import cv2
import matplotlib.pyplot as plt
import skimage.filters as skf

# Read image from file in gray scale mode and  convert to numpy array
im = cv2.imread("11.png", cv2.IMREAD_GRAYSCALE)
img = cv2.normalize(im.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)

# Canny
EdgeC = cv2.Canny(im, 100, 200)

# Sobel
EdgeS = skf.sobel(img)
# EdgeSx = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
# EdgeSy = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
# EdgeS = EdgeSx + EdgeSy

# Prewitt
EdgeP = skf.prewitt(img)
# kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
# kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
# EdgePx = cv2.filter2D(img, -1, kernelx)
# EdgePy = cv2.filter2D(img, -1, kernely)
# EdgeP = EdgePx + EdgePy

# Roberts
EdgeR = skf.roberts(img)

plt.subplot(221)
plt.imshow(EdgeC, cmap='gray')
plt.title("Canny filter")
plt.subplot(222)
plt.imshow(EdgeS, cmap='gray')
plt.title("Sobel filter")
plt.subplot(223)
plt.imshow(EdgeP, cmap='gray')
plt.title("Prewitt filter")
plt.subplot(224)
plt.imshow(EdgeR, cmap='gray')
plt.title("Roberts filter")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
