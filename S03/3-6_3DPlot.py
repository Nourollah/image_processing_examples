import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

img = np.zeros([256, 256]).astype("float")
img[108:148, 108:148] = 1
plt.subplot(131)
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
imgF = np.fft.fft2(img)
IMgFD = np.abs(imgF)
Max = np.max(np.max(IMgFD))
plt.subplot(132)
plt.imshow(IMgFD / Max, cmap='gray', vmin=0, vmax=1)

imgFShift = np.fft.fftshift(imgF)
IMgFD2 = np.abs(imgFShift)
Max2 = np.max(np.max(IMgFD2))
plt.subplot(133)
plt.imshow(IMgFD2 / Max2, cmap='gray', vmin=0, vmax=1)
#_______________________________________________________________3D Plot
fig4 = plt.figure()
ax = fig4.gca(projection='3d')
IMgFD2_2 = np.meshgrid(np.abs(IMgFD2))
ax = fig4.add_subplot(111, projection="3d")
x = np.zeros(256)
y = np.zeros(256)
ax.plot_surface(x, y, IMgFD2_2, cmap=cm.coolwarm, linewidth=1, antialiased=False)
plt.show()

