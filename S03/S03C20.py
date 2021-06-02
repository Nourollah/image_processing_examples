import cv2
import numpy as np
import matplotlib.pyplot as plt

# Use gray for plots color map
plt.gray()


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def gaussian(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = np.exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
    return base

# Read image in gray scale mode
im = cv2.normalize(cv2.imread("Natural.png", cv2.IMREAD_GRAYSCALE).astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
# Plot original image
plt.subplot(131)
plt.imshow(im)
plt.title("Img")
# Save image shape into m and n
[m, n] = im.shape
# Prevent to divide on zero
f = im + 0.0001
# Take log from img
z = np.log(f)
# Going to frequency mode
Z = np.fft.fftshift(np.fft.fft2(z))
# Set dim and sigma
v1 = m
v2 = n
v = 15
# Create gaussian mask
H = gaussian(v, (v1, v2))
# Apply mask to image
REF = Z * (1 - H)
LUM = Z * H
# Back to place from frequency
reflect = np.exp(np.fft.ifft2(np.fft.ifftshift(REF)))
lum = np.exp(np.fft.ifft2(np.fft.ifftshift(LUM)))
IFFTt = np.abs(reflect)
Ref = IFFTt
IFFTt2 = np.abs(lum)
Illu = IFFTt2
# Plot images
plt.subplot(132)
plt.imshow(Illu)
plt.title("Ill")
plt.subplot(133)
plt.imshow(Ref)
plt.title("Ref")
plt.show()
