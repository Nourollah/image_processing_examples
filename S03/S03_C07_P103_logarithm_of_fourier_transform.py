import numpy as np
import matplotlib.pyplot as plt

# Create array initializing with zeros
img = np.zeros([256, 256])
# Change some pixel to absolute white
img[108:148, 108:148] = 1
# Going to frequency
imgF = np.fft.fftshift(np.fft.fft2(img))
# Take real part of image complex numbers
IMgFd = np.abs(imgF)
# Take log from image
IMGLog = np.log(1 + IMgFd)
# Calc max of image
Max = np.max(IMGLog)
# Plot image
plt.imshow(IMGLog / Max, cmap='gray', vmin=0, vmax=1)
plt.show()
