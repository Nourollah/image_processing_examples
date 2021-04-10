import numpy as np

# Define array as image
img = np.arange(1, 10).reshape(3, 3)
# Apply 2D Fourier transform to img with default and custom shape
imgF1 = np.fft.fft2(img)
imgF2 = np.fft.fft2(img, [5, 4])

